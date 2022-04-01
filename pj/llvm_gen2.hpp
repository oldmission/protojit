#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace llvm {
struct TargetMachine;
}

namespace pj {
std::unique_ptr<mlir::Pass> createLLVMGenPass(const llvm::TargetMachine*);
}  // namespace pj
