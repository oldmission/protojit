#pragma once

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace llvm {
struct TargetMachine;
}  // namespace llvm

namespace pj {
// New pipeline
std::unique_ptr<mlir::Pass> createIRGenPass();
std::unique_ptr<mlir::Pass> createLLVMGenPass(const llvm::TargetMachine*);

// Legacy pipeline
std::unique_ptr<mlir::Pass> createInlineRegionsPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
}  // namespace pj
