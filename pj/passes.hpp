#pragma once

#include <memory>

namespace mlir {
class Pass;
}  // namespace mlir

namespace llvm {
struct TargetMachine;
class FunctionPass;
class DataLayout;
}  // namespace llvm

namespace pj {
std::unique_ptr<mlir::Pass> createIRGenPass();
std::unique_ptr<mlir::Pass> createGenSizeFunctionsPass();
std::unique_ptr<mlir::Pass> createLLVMGenPass(const llvm::TargetMachine*);
}  // namespace pj

namespace llvm {
FunctionPass* createCopyCoalescingPass(const llvm::TargetMachine& layout);
}  // namespace llvm
