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
#include <llvm/Support/TargetSelect.h>
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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include <cstring>

#include "ir.hpp"
#include "llvm_extra.hpp"
#include "passes.hpp"
#include "plan.hpp"
#include "portal.hpp"
#include "portal_impl.hpp"
#include "protojit.hpp"
#include "reflect.hpp"
#include "runtime.h"
#include "util.hpp"

#include "schema/precompile.hpp"

// #define DISABLE_LIBC

extern "C" {
// Generate definitions for the precompiled methods that will be used in the
// bootstrapping phase when the precompiled methods are being compiled, and
// subsequently overriden when the precompiled methods are linked in. If the
// precompiled methods are not statically linked in, attempts to find them
// dynamically using dlsym().
#define WEAK_DEFINITIONS(V, _)                                                \
  size_t __attribute__((weak))                                                \
      EXTERN_GET_PROTO_SIZE(V)(const pj::reflect::Protocol* proto) {          \
    static std::once_flag flag;                                               \
    static pj::SizeFunction<pj::reflect::Protocol> size;                      \
    std::call_once(flag, []() {                                               \
      size = reinterpret_cast<pj::SizeFunction<pj::reflect::Protocol>>(       \
          dlsym(RTLD_NEXT, STR(EXTERN_GET_PROTO_SIZE(V))));                   \
    });                                                                       \
    return size(proto);                                                       \
  }                                                                           \
  void __attribute__((weak))                                                  \
      EXTERN_ENCODE_PROTO(V)(const pj::reflect::Protocol* proto, char* buf) { \
    static std::once_flag flag;                                               \
    static pj::EncodeFunction<pj::reflect::Protocol> encode;                  \
    std::call_once(flag, []() {                                               \
      encode = reinterpret_cast<pj::EncodeFunction<pj::reflect::Protocol>>(   \
          dlsym(RTLD_NEXT, STR(EXTERN_ENCODE_PROTO(V))));                     \
    });                                                                       \
    encode(proto, buf);                                                       \
  }                                                                           \
  BoundedBuffer __attribute__((weak)) EXTERN_DECODE_PROTO(V)(                 \
      const char* msg, pj::reflect::Protocol* result, BoundedBuffer buffer,   \
      pj::DecodeHandler<pj::reflect::Protocol, void> handlers[],              \
      void* state) {                                                          \
    static std::once_flag flag;                                               \
    static pj::DecodeFunction<pj::reflect::Protocol, void> decode;            \
    std::call_once(flag, []() {                                               \
      decode =                                                                \
          reinterpret_cast<pj::DecodeFunction<pj::reflect::Protocol, void>>(  \
              dlsym(RTLD_NEXT, STR(EXTERN_DECODE_PROTO(V))));                 \
    });                                                                       \
    return decode(msg, result, buffer, handlers, state);                      \
  }

FOR_EACH_COMPATIBLE_VERSION(WEAK_DEFINITIONS)

#undef WEAK_DEFINITIONS
}  // extern "C"

namespace pj {

using namespace ir;

std::once_flag g_has_initialized_llvm;

void initializeLLVM() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

#ifndef NDEBUG
  const char* env = getenv("PROTOJIT_INTERNAL_DEBUG_TYPES");
  if (env != nullptr && env[0] != '\0') {
    if (!std::strcmp(env, "ON") || !std::strcmp(env, "1")) {
      llvm::DebugFlag = 1;
    } else {
      std::istringstream str(env);
      std::string type;
      std::vector<std::string> types;
      std::vector<const char*> types_p;
      while (getline(str, type, ',')) {
        types.push_back(type);
      }

      for (const std::string& type : types) {
        types_p.push_back(type.c_str());
      }

      if (types.size() > 0) {
        llvm::DebugFlag = 1;
        llvm::setCurrentDebugTypes(types_p.data(), types_p.size());
      }
    }
  }
#endif
}

ProtoJitContext::ProtoJitContext() : builder_(&ctx_) {
  std::call_once(g_has_initialized_llvm, initializeLLVM);

  resetModule();
  ctx_.getOrLoadDialect<ProtoJitDialect>();
}

ProtoJitContext::~ProtoJitContext() {}

void ProtoJitContext::resetModule() {
  module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
}

uint64_t ProtoJitContext::getProtoSize(types::ProtocolType proto) {
  llvm::BumpPtrAllocator alloc;
  reflect::Protocol reflected = reflect::reflect(alloc, proto);

  // The total size is encoded in 8 bytes in the beginning.
  uint64_t size = 8;

  // Each version contributes 8 bytes for the version number, 8 bytes for the
  // length, and one copy of the schema in that version's format.
#define GET_SIZE_FOR_VERSION(V, _) \
  size += 8;                       \
  size += 8;                       \
  size += schema::GET_PROTO_SIZE(V)(&reflected);
  FOR_EACH_COMPATIBLE_VERSION(GET_SIZE_FOR_VERSION)
#undef GET_SIZE_FOR_VERSION

  return size;
}

void ProtoJitContext::encodeProto(types::ProtocolType proto, char* buf) {
  llvm::BumpPtrAllocator alloc;
  reflect::Protocol reflected = reflect::reflect(alloc, proto);

  char* buf_start = buf;
  buf += 8;

#define ENCODE_FOR_VERSION(V, N)                                         \
  {                                                                      \
    uint64_t version = N;                                                \
    uint64_t size = RoundUp(schema::GET_PROTO_SIZE(V)(&reflected), 8ul); \
    std::memcpy(&buf[0], &version, 8);                                   \
    std::memcpy(&buf[8], &size, 8);                                      \
    schema::ENCODE_PROTO(V)(&reflected, &buf[16]);                       \
    buf += 16 + size;                                                    \
  }
  FOR_EACH_COMPATIBLE_VERSION(ENCODE_FOR_VERSION)
#undef ENCODE_FOR_VERSION

  // Encode the size in the first 8 bytes.
  uint64_t total_size = buf - buf_start;
  std::memcpy(buf_start, &total_size, 8);
}

types::ProtocolType ProtoJitContext::decodeProto(const char* buf) {
  reflect::Protocol proto;

  struct {
    uint64_t version;
    uint64_t size;
    const char* data;
    BoundedBuffer (*decode_fn)(const char*, reflect::Protocol*, BoundedBuffer);
  } latest_compatible = {};

  uint64_t size;
  std::memcpy(&size, &buf[0], 8);

  const char* buf_end = buf + size;
  buf += 8;

  while (buf < buf_end) {
    uint64_t version;
    uint64_t size;

    std::memcpy(&version, &buf[0], 8);
    std::memcpy(&size, &buf[8], 8);

#define MATCH_VERSION(V, N)                                 \
  else if (version == N && N > latest_compatible.version) { \
    latest_compatible.version = N;                          \
    latest_compatible.size = size;                          \
    latest_compatible.data = &buf[16];                      \
    latest_compatible.decode_fn = &schema::DECODE_PROTO(V); \
  }

    if (false)
      ;
    FOR_EACH_COMPATIBLE_VERSION(MATCH_VERSION)
#undef MATCH_VERSION
    else {
      // None of the versions coming later in the file will match because either
      //  (1) The major version is different, so entirely incompatible.
      //  (2) The major version is the same, but the versions come in
      //      monotonically increasing order, so any upcoming version will also
      //      be incompatible with the current minor version.
      break;
    }

    buf += 16 + size;
  }

  if (latest_compatible.version == 0) {
    return {};
  }

  // Decode the latest compatible version found in the file.
  for (size_t dec_size = 8192;; dec_size *= 2) {
    auto dec_buffer = std::make_unique<char[]>(dec_size);

    auto bbuf = latest_compatible.decode_fn(
        latest_compatible.data, &proto,
        {.ptr = dec_buffer.get(), .size = dec_size});

    if (bbuf.ptr != nullptr) {
      auto wire = types::WireDomainAttr::unique(&ctx_);
      return reflect::unreflect(proto, ctx_, wire).cast<types::ProtocolType>();
    }
  }
}

void ProtoJitContext::addEncodeFunction(std::string_view name, mlir::Type src,
                                        types::ProtocolType protocol,
                                        llvm::StringRef src_path) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<EncodeFunctionOp>(
      loc, std::string{name}, src, protocol,
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
      loc, std::string{name}, src, dst,
      mlir::ArrayAttr::get(&ctx_, handler_attrs)));
}

void ProtoJitContext::addSizeFunction(std::string_view name, mlir::Type src,
                                      types::ProtocolType protocol,
                                      llvm::StringRef src_path, bool round_up) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<SizeFunctionOp>(
      loc, std::string{name}, src, protocol,
      types::PathAttr::fromString(&ctx_, src_path), round_up));
}

void ProtoJitContext::addProtocolDefinition(std::string_view name,
                                            std::string_view size_name,
                                            llvm::StringRef proto_data) {
  // TODO: use a more interesting location
  auto loc = builder_.getUnknownLoc();
  module_->push_back(builder_.create<DefineProtocolOp>(
      loc, std::string{name}, std::string{size_name}, proto_data));
}

#define DEBUG_TYPE "pj.compile"

static std::unique_ptr<llvm::TargetMachine> getTargetMachine(bool pic) {
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

  // Must generate position-independent code for the precompiled schema code
  // because it is being linked into libprotojit.so, a shared library.
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      target_triple, cpu, features.getString(), {},
      pic ? llvm::Reloc::Model::PIC_ : llvm::Reloc::Model::Static));
  if (!machine) {
    throw InternalError("Cannot create LLVM target machine!");
  }

  return machine;
}

std::pair<std::unique_ptr<llvm::LLVMContext>, std::unique_ptr<llvm::Module>>
ProtoJitContext::compileToLLVM(bool pic, size_t opt_level) {
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

  auto machine = getTargetMachine(pic);

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

void ProtoJitContext::precompile(std::string_view filename, bool pic,
                                 size_t opt_level) {
  auto [llvm_context, llvm_module] = compileToLLVM(pic, opt_level);

  std::error_code EC;
  llvm::raw_fd_ostream dest(filename, EC, llvm::sys::fs::OF_None);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    abort();
  }

  llvm::legacy::PassManager asm_pass;
  auto machine = getTargetMachine(pic);
  machine->addPassesToEmitFile(asm_pass, dest, nullptr,
                               llvm::CodeGenFileType::CGFT_ObjectFile);
  asm_pass.run(*llvm_module);
}

std::unique_ptr<Portal> ProtoJitContext::compile(size_t opt_level) {
  auto [llvm_context, llvm_module] = compileToLLVM(false, opt_level);

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
