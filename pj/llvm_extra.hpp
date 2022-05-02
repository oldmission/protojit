#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/DialectConversion.h>

#define GET_OP_CLASSES
#include "pj/llvm_extra.hpp.inc"

namespace mlir {
namespace LLVM {

class LLVMExtraDialect : public Dialect {
 public:
  explicit LLVMExtraDialect(MLIRContext* ctx);
  ~LLVMExtraDialect();
  static llvm::StringRef getDialectNamespace() { return "llvm_extra"; }
};

}  // namespace LLVM

void registerLLVMExtraDialectTranslation(DialectRegistry& registry);

}  // namespace mlir
