#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Target/LLVMIR/LLVMTranslationInterface.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "llvm_extra.hpp"

namespace mlir {
namespace LLVM {

LLVMExtraDialect::~LLVMExtraDialect() {}

LLVMExtraDialect::LLVMExtraDialect(MLIRContext* ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<LLVMExtraDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "pj/llvm_extra.cpp.inc"
      >();
}

}  // namespace LLVM

namespace {
class LLVMExtraDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
 public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult convertOperation(
      Operation* op, llvm::IRBuilderBase& builder,
      LLVM::ModuleTranslation& moduleTranslation) const final {
    auto& opInst = *op;
#include "pj/llvm_extra_conversions.cpp.inc"
    return success();
  }
};
}  // end namespace

void registerLLVMExtraDialectTranslation(DialectRegistry& registry) {
  registry.insert<LLVM::LLVMExtraDialect>();
  registry.addDialectInterface<LLVM::LLVMExtraDialect,
                               LLVMExtraDialectLLVMIRTranslationInterface>();
}
}  // namespace mlir

#define GET_OP_CLASSES
#include "pj/llvm_extra.cpp.inc"
