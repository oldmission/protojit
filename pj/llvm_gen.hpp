#pragma once

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <memory>

namespace mlir {
class Pass;
}

namespace pj {
namespace ir {

// Convert ProtoJit IR types to LLVM.
class ProtoJitTypeConverter : public mlir::LLVMTypeConverter {
 public:
  ProtoJitTypeConverter(mlir::MLIRContext* C);
};

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

}  // namespace ir
}  // namespace pj
