#include "context.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCSymbol.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetRegistry.h>

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

#include "ir.hpp"
#include "passes.hpp"
#include "portal.hpp"
#include "portal_impl.hpp"

namespace pj {
using namespace ir2;

ProtoJitContext::ProtoJitContext()
    : builder_(&ctx_),
      module_(mlir::ModuleOp::create(builder_.getUnknownLoc())) {
  ctx_.getOrLoadDialect<pj::ir::ProtoJitDialect>();
  ctx_.getOrLoadDialect<mlir::StandardOpsDialect>();
  ctx_.getOrLoadDialect<mlir::scf::SCFDialect>();
}

ProtoJitContext::~ProtoJitContext() {}

void ProtoJitContext::addEncodeFunction(std::string_view name, mlir::Type src,
                                        llvm::StringRef src_path,
                                        types::ProtocolType protocol) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<EncodeFunctionOp>(
      loc, name, src, protocol, types::PathAttr::fromString(&ctx_, src_path)));
}

void ProtoJitContext::addDecodeFunction(
    std::string_view name, types::ProtocolType src, mlir::Type dst,
    const std::vector<std::pair<std::string, const void*>>& handlers) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();

  llvm::SmallVector<mlir::Attribute> handler_attrs;
  for (const auto& hand : handlers) {
    handler_attrs.push_back(types::DispatchHandlerAttr::get(
        &ctx_, types::PathAttr::fromString(&ctx_, hand.first), hand.second));
  }

  module_->push_back(builder_.create<DecodeFunctionOp>(
      loc, name, src, dst, mlir::ArrayAttr::get(&ctx_, handler_attrs)));
}

void ProtoJitContext::addSizeFunction(std::string_view name, mlir::Type src,
                                      llvm::Optional<llvm::StringRef> src_path,
                                      mlir::Type dst) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<SizeFunctionOp>(
      loc, name, src,
      src_path ? types::PathAttr::fromString(&ctx_, *src_path)
               : types::PathAttr{},
      dst));
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

std::unique_ptr<Portal> ProtoJitContext::compile(bool new_pipeline) {
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

  mlir::PassManager pjpm(&ctx_);
  pjpm.addPass(mlir::createLowerToCFGPass());
  pjpm.addPass(pj::createInlineRegionsPass());
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
  if (new_pipeline) {
    pm.addPass(pj::createLLVMGenPass(machine.get()));
  } else {
    pm.addPass(createLowerToLLVMPass());
  }
  pm.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*module_))) {
    throw InternalError("Error lowering to LLVM.");
  }

  LLVM_DEBUG(
      llvm::errs() << "===================================================\n"
                      "After LLVM lowering:\n"
                      "==================================================\n";
      module_->dump());

  std::unique_ptr<llvm::LLVMContext> llvm_context(new llvm::LLVMContext());
  mlir::registerLLVMDialectTranslation(*module_->getContext());
  auto llvm_module = mlir::translateModuleToLLVMIR(*module_, *llvm_context);
  if (!llvm_module) {
    throw InternalError("Failed to emit LLVM IR");
  }

  mlir::ExecutionEngine::setupTargetTriple(llvm_module.get());

  auto opt_pipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = opt_pipeline(llvm_module.get())) {
    std::string msg;
    llvm::raw_string_ostream str(msg);
    str << "Failed to optimize LLVM IR " << err << "\n";
    throw InternalError(std::move(msg));
  }

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After LLVM optimization:\n"
                      "==================================================\n";
      llvm::errs() << *llvm_module << "\n");

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

  LLVM_DEBUG({
    auto Filename = "output.o";
    std::error_code EC;
    llvm::raw_fd_ostream dest(Filename, EC, llvm::sys::fs::OF_None);

    if (EC) {
      llvm::errs() << "Could not open file: " << EC.message();
      abort();
    }

    llvm::legacy::PassManager asm_pass;
    machine->addPassesToEmitFile(asm_pass, dest, nullptr,
                                 llvm::CodeGenFileType::CGFT_ObjectFile);
    asm_pass.run(*llvm_module);
  });

  llvm::raw_svector_ostream dest(out);

  llvm::legacy::PassManager pass;
  machine->addPassesToEmitFile(pass, dest, nullptr,
                               llvm::CodeGenFileType::CGFT_ObjectFile);
  pass.run(*llvm_module);

  auto jit = llvm::orc::LLJITBuilder().create();

  if (!jit) abort();

  if (auto err = jit.get()->addIRModule(llvm::orc::ThreadSafeModule(
          std::move(llvm_module), std::move(llvm_context)))) {
    abort();
  }

  auto buffer = std::make_unique<llvm::SmallVectorMemoryBuffer>(std::move(out));

  auto loaded_object =
      llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());

  if (!loaded_object) {
    throw InternalError("Failed to generate machine code.");
  }

  return std::make_unique<PortalImpl>(std::move(*jit));
}

}  // namespace pj
