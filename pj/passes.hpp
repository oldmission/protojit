#pragma once

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace llvm {
struct TargetMachine;
}  // namespace llvm

namespace pj {
std::unique_ptr<mlir::Pass> createIRGenPass();
std::unique_ptr<mlir::Pass> createLLVMGenPass(const llvm::TargetMachine*);
}  // namespace pj
