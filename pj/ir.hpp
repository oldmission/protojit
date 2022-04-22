#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>

#include "exceptions.hpp"
#include "types.hpp"

#define GET_OP_CLASSES
#include "pj/enums.hpp.inc"
#include "pj/ir.hpp.inc"

namespace pj {
namespace ir {

using namespace mlir;

class ProtoJitDialect : public Dialect {
 public:
  explicit ProtoJitDialect(MLIRContext* ctx);
  ~ProtoJitDialect();

  static llvm::StringRef getDialectNamespace() { return "pj"; }

  void printType(Type type, DialectAsmPrinter& printer) const override;
  void printAttribute(Attribute type,
                      DialectAsmPrinter& printer) const override;
};

void printAttrForFunctionName(llvm::raw_ostream& os, mlir::Attribute attr);

}  // namespace ir
}  // namespace pj
