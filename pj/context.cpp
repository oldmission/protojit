#include "context.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCSymbol.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>

#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <cstring>

#include "ir.hpp"
#include "llvm_extra.hpp"
#include "passes.hpp"
#include "plan.hpp"
#include "portal.hpp"
#include "portal_impl.hpp"
#include "protojit.hpp"
#include "reflect.hpp"
#include "span.hpp"

// #define DISABLE_LIBC

namespace pj {
using namespace ir;

ProtoJitContext::ProtoJitContext() : builder_(&ctx_) {
  resetModule();
  ctx_.getOrLoadDialect<ProtoJitDialect>();
}

ProtoJitContext::~ProtoJitContext() {}

void ProtoJitContext::resetModule() {
  module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
}

Portal* ProtoJitContext::schemaPortal() {
  if (schema_portal_) {
    return schema_portal_.get();
  }

  PJContext* ctx = reinterpret_cast<PJContext*>(&ctx_);
  auto schema_host = mlir::Type::getFromOpaquePointer(
      gen::BuildPJType<reflect::Protocol>::build(ctx));

  // TODO: This is temporary for testing! In the future this will be in a
  // reflect.version.pj.hpp file which will be generated by sourcegen
  auto schema_wire =
      plan_protocol(ctx_, schema_host, types::PathAttr::none(&ctx_));

  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<EncodeFunctionOp>(
      loc, "schema_encode", schema_host, schema_wire,
      types::PathAttr::none(&ctx_)));
  module_->push_back(builder_.create<DecodeFunctionOp>(
      loc, "schema_decode", schema_wire, schema_host,
      mlir::ArrayAttr::get(&ctx_, {})));
  module_->push_back(builder_.create<SizeFunctionOp>(
      loc, "schema_size", schema_host, schema_wire,
      types::PathAttr::none(&ctx_), false));

  schema_portal_ = compile(0);
  return schema_portal_.get();
}

uint64_t ProtoJitContext::getProtoSize(types::ProtocolType proto) {
  llvm::BumpPtrAllocator alloc;
  reflect::Protocol reflected = reflect::reflect(alloc, proto);

  const auto size_fn =
      schemaPortal()->GetSizeFunction<reflect::Protocol>("schema_size", true);
  assert(size_fn);
  return size_fn(&reflected);
}

void ProtoJitContext::encodeProto(types::ProtocolType proto, char* buf) {
  llvm::BumpPtrAllocator alloc;
  reflect::Protocol reflected = reflect::reflect(alloc, proto);

  const auto encode_fn = schemaPortal()->GetEncodeFunction<reflect::Protocol>(
      "schema_encode", true);
  assert(encode_fn);
  encode_fn(&reflected, buf);
}

types::ProtocolType ProtoJitContext::decodeProto(const char* buf) {
  reflect::Protocol reflected;

  const auto decode_fn = schemaPortal()->GetDecodeFunction<reflect::Protocol>(
      "schema_decode", true);
  assert(decode_fn);

  uint64_t dec_size = 8192;
  while (true) {
    auto dec_buffer = std::make_unique<char[]>(dec_size);
    auto bbuf =
        decode_fn(buf, &reflected, {.ptr = dec_buffer.get(), .size = dec_size},
                  nullptr, nullptr);
    if (bbuf.ptr == nullptr) {
      dec_size *= 2;
      continue;
    }
    return reflect::unreflect(reflected, ctx_).cast<types::ProtocolType>();
  }
}

void ProtoJitContext::addEncodeFunction(std::string_view name, mlir::Type src,
                                        types::ProtocolType protocol,
                                        llvm::StringRef src_path) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<EncodeFunctionOp>(
      loc, kUserFunctionPrefix + std::string{name}, src, protocol,
      types::PathAttr::fromString(&ctx_, src_path)));
}

void ProtoJitContext::addDecodeFunction(
    std::string_view name, types::ProtocolType src, mlir::Type dst,
    const std::vector<std::string>& handlers) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();

  llvm::SmallVector<mlir::Attribute> handler_attrs;
  for (const auto& hand : llvm::enumerate(handlers)) {
    handler_attrs.push_back(types::DispatchHandlerAttr::get(
        &ctx_, types::PathAttr::fromString(&ctx_, hand.value()), hand.index()));
  }

  module_->push_back(builder_.create<DecodeFunctionOp>(
      loc, kUserFunctionPrefix + std::string{name}, src, dst,
      mlir::ArrayAttr::get(&ctx_, handler_attrs)));
}

void ProtoJitContext::addSizeFunction(std::string_view name, mlir::Type src,
                                      types::ProtocolType protocol,
                                      llvm::StringRef src_path, bool round_up) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<SizeFunctionOp>(
      loc, kUserFunctionPrefix + std::string{name}, src, protocol,
      types::PathAttr::fromString(&ctx_, src_path), round_up));
}

#define DEBUG_TYPE "pj.compile"

static std::unique_ptr<llvm::TargetMachine> getTargetMachine() {
  // Setup the machine properties from the current architecture.
  auto target_triple = llvm::sys::getDefaultTargetTriple();
  std::string error_message;
  auto* target =
      llvm::TargetRegistry::lookupTarget(target_triple, error_message);
  if (!target) {
    throw InternalError("Cannot find LLVM target (" + error_message + ")!");
  }

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> host_features;

  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto& f : host_features) {
      features.AddFeature(f.first(), f.second);
    }
  }

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      target_triple, cpu, features.getString(), {}, {}));
  if (!machine) {
    throw InternalError("Cannot create LLVM target machinue!");
  }

  return machine;
}

std::pair<std::unique_ptr<llvm::LLVMContext>, std::unique_ptr<llvm::Module>>
ProtoJitContext::compileToLLVM(size_t opt_level) {
  ctx_.getOrLoadDialect<mlir::LLVM::LLVMExtraDialect>();
  ctx_.getOrLoadDialect<mlir::StandardOpsDialect>();
  ctx_.getOrLoadDialect<mlir::scf::SCFDialect>();

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "Before compilation:\n"
                      "==================================================\n";
      module_->dump());

  mlir::PassManager pj_lower(&ctx_);
  pj_lower.addPass(pj::createIRGenPass());

  if (mlir::failed(pj_lower.run(*module_))) {
    throw InternalError("Error generating PJIR.");
  }

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After PJIR generation:\n"
                      "==================================================\n";
      module_->dump());

  mlir::PassManager pj_size(&ctx_);
  pj_size.addPass(pj::createGenSizeFunctionsPass());

  if (mlir::failed(pj_size.run(*module_))) {
    throw InternalError("Error generating PJ size functions.");
  }

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After PJ size function generation:\n"
                      "==================================================\n";
      module_->dump());

  mlir::PassManager pjpm(&ctx_);
  pjpm.addPass(mlir::createLowerToCFGPass());
  pjpm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pjpm.run(*module_))) {
    throw InternalError("Error optimizing PJIR.");
  }

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After PJIR optimization:\n"
                      "==================================================\n";
      module_->dump());

  auto machine = getTargetMachine();

  // Lower to LLVM IR
  mlir::PassManager pm(&ctx_);
  pm.addPass(pj::createLLVMGenPass(machine.get()));

  if (mlir::failed(pm.run(*module_))) {
    throw InternalError("Error lowering to LLVM.");
  }

  LLVM_DEBUG(
      llvm::errs() << "===================================================\n"
                      "After LLVM lowering:\n"
                      "==================================================\n";
      module_->dump());

  std::unique_ptr<llvm::LLVMContext> llvm_context(new llvm::LLVMContext());

  DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerLLVMExtraDialectTranslation(registry);
  module_->getContext()->appendDialectRegistry(registry);

  auto llvm_module = mlir::translateModuleToLLVMIR(*module_, *llvm_context);
  if (!llvm_module) {
    throw InternalError("Failed to emit LLVM IR");
  }

  // Clear the module for future compilations using this context.
  resetModule();

  mlir::ExecutionEngine::setupTargetTriple(llvm_module.get());

  {
    // See mlir/lib/ExecutionEngine/OptUtils.cpp's populatePassManagers for the
    // origin of the following choices.
    llvm::legacy::PassManager modulePM;
    llvm::legacy::FunctionPassManager funcPM(llvm_module.get());

    modulePM.add(llvm::createTargetTransformInfoWrapperPass(
        machine.get()->getTargetIRAnalysis()));
    funcPM.add(llvm::createTargetTransformInfoWrapperPass(
        machine.get()->getTargetIRAnalysis()));

    llvm::PassManagerBuilder builder;
    builder.OptLevel = opt_level;
    builder.Inliner = llvm::createFunctionInliningPass(
        /*optLevel=*/opt_level, /*sizeLevel=*/0,
        /*DisableInlineHotCallSite=*/false);
    builder.LoopVectorize = true;
    builder.SLPVectorize = true;

#ifdef DISABLE_LIBC
    // Set TargetLibraryInfoWrapperPass to assume that no library functions are
    // available.
    auto TLII = new llvm::TargetLibraryInfoImpl{
        llvm::Triple{llvm_module->getTargetTriple()}};
    TLII->disableAllFunctions();
    builder.LibraryInfo = TLII;
#endif

    builder.populateModulePassManager(modulePM);
    builder.populateFunctionPassManager(funcPM);

    funcPM.doInitialization();
    for (auto& func : *llvm_module) {
      funcPM.run(func);
    }
    funcPM.doFinalization();
    modulePM.run(*llvm_module);
  }

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After LLVM optimization (preliminary):\n"
                      "==================================================\n";
      llvm::errs() << *llvm_module << "\n");

  if (opt_level >= 3) {
    {
      llvm::legacy::FunctionPassManager funcPM(llvm_module.get());
      funcPM.doInitialization();
      funcPM.add(llvm::createCopyExtendingPass(*machine));
      for (auto& func : *llvm_module) {
        funcPM.run(func);
      }
      funcPM.doFinalization();
    }

    LLVM_DEBUG(llvm::errs()
                   << "==================================================\n"
                      "After LLVM optimization (extending):\n"
                      "==================================================\n";
               llvm::errs() << *llvm_module << "\n");

    {
      llvm::legacy::FunctionPassManager funcPM(llvm_module.get());
      funcPM.doInitialization();
      funcPM.add(llvm::createCopyCoalescingPass(*machine));
      funcPM.add(llvm::createCFGSimplificationPass());
      funcPM.add(llvm::createEarlyCSEPass());
      funcPM.add(llvm::createAggressiveDCEPass());
      for (auto& func : *llvm_module) {
        funcPM.run(func);
      }
      funcPM.doFinalization();
    }

    LLVM_DEBUG(llvm::errs()
                   << "==================================================\n"
                      "After LLVM optimization (coalescing):\n"
                      "==================================================\n";
               llvm::errs() << *llvm_module << "\n");
  }

  llvm::SmallVector<char, 0x1000> out;

  // Print assembly as well when debugging.
  LLVM_DEBUG({
    llvm::raw_svector_ostream asm_dest(out);

    llvm::legacy::PassManager asm_pass;
    machine->addPassesToEmitFile(asm_pass, asm_dest, nullptr,
                                 llvm::CodeGenFileType::CGFT_AssemblyFile);
    asm_pass.run(*llvm_module);

    llvm::errs() << "==================================================\n"
                    "Assembly:\n"
                    "==================================================\n";
    llvm::errs() << out;

    out.clear();
  });

  return std::make_pair(std::move(llvm_context), std::move(llvm_module));
}

void ProtoJitContext::precompile(std::string_view filename, size_t opt_level) {
  auto [llvm_context, llvm_module] = compileToLLVM(opt_level);

  std::error_code EC;
  llvm::raw_fd_ostream dest(filename, EC, llvm::sys::fs::OF_None);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    abort();
  }

  llvm::legacy::PassManager asm_pass;
  auto machine = getTargetMachine();
  machine->addPassesToEmitFile(asm_pass, dest, nullptr,
                               llvm::CodeGenFileType::CGFT_ObjectFile);
  asm_pass.run(*llvm_module);
}

std::unique_ptr<Portal> ProtoJitContext::compile(size_t opt_level) {
  auto [llvm_context, llvm_module] = compileToLLVM(opt_level);

  auto jit = []() {
    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit) abort();
    return std::move(jit.get());
  }();

#ifndef DISABLE_LIBC
  llvm::orc::SymbolMap symbols;
  symbols.try_emplace(jit->mangleAndIntern("memcpy"),
                      llvm::JITEvaluatedSymbol::fromPointer(&memcpy));
  if (auto err =
          jit->getMainJITDylib().define(llvm::orc::absoluteSymbols(symbols))) {
    abort();
  }
#endif

  if (auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(
          std::move(llvm_module), std::move(llvm_context)))) {
    abort();
  }

  return std::make_unique<PortalImpl>(std::move(jit));
}

}  // namespace pj
