#pragma once

#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringMap.h>
#include <mlir/Support/LLVM.h>

#include <llvm/Support/raw_ostream.h>

#include <memory>

#include "span.hpp"

namespace mlir {
class Operation;
}  // namespace mlir

namespace pj {

// Some operations in ProtoJIT IR have side-effects which, when lowered, need
// access to external state to implement the effect. This is achieved by passing
// pointers to the state down the call-stack to the operations which require it.
//
// This analysis identifies functions whose parameters need to be updated to
// accomodate these pointers and call-sites which need to pass them through.
class SideEffectAnalysis {
 public:
  explicit SideEffectAnalysis(mlir::Operation* root);

  bool hasEffects(mlir::Operation* op) const {
    return effect_functions.contains(op) or effect_points.contains(op);
  }

  Span<size_t> flattenedBufferArguments(llvm::StringRef callee) const;

  mlir::Operation* effectProviderFor(mlir::Operation* op) const {
    if (auto it = effect_providers.find(op); it != effect_providers.end()) {
      return it->second;
    }
    return nullptr;
  }

  using OpSet =
      llvm::DenseSet<mlir::Operation*, llvm::DenseMapInfo<mlir::Operation*>>;

 private:
  OpSet effect_points;
  OpSet effect_functions;
  llvm::DenseMap<mlir::Operation*, mlir::Operation*> effect_providers;
  llvm::StringMap<llvm::SmallVector<size_t, 1>> flattened_buffer_args;
};

}  // namespace pj
