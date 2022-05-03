#include <deque>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include "ir.hpp"
#include "side_effect_analysis.hpp"
#include "span.hpp"

namespace pj {

using namespace mlir;
using namespace ir;
using namespace types;

SideEffectAnalysis::SideEffectAnalysis(Operation* root) {
  // Most functions in ProtoJIT IR have only one call site.
  DenseMap<Operation*, SmallVector<Operation*, 1>> call_sites_per_fn;
  DenseMap<Operation*, Operation*> enclosing_fns;
  std::deque<Operation*> roots;

  ModuleOp mod = cast<ModuleOp>(root);

  // 1. Build bipartite graph of exception points (calls + roots)
  //    and functions.
  //
  //    Record root points (SetCallbackOp, TODO: ThrowOp as well).

  root->walk([&](Operation* op) {
    if (!isa<SetCallbackOp>(op) && !isa<CallOp>(op) &&
        !isa<InvokeCallbackOp>(op)) {
      return;
    }

    if (isa<DecodeCatchOp>(op->getParentOp()) ||
        isa<SizeOp>(op->getParentOp())) {
      // Effect doesn't propagate.
    } else {
      assert(isa<FuncOp>(op->getParentOp()));
      enclosing_fns[op] = op->getParentOp();
    }
    effect_providers[op] = op->getParentOp();

    if (auto call = dyn_cast<CallOp>(op)) {
      // TODO: lookupSymbol is a linear scan
      call_sites_per_fn[mod.lookupSymbol(call.callee())].push_back(op);
    } else {
      roots.push_back(op);
    }
  });

  // 2. Implement closure via BFS.

  while (!roots.empty()) {
    auto* effect_op = roots.front();
    if (auto it = enclosing_fns.find(effect_op); it != enclosing_fns.end()) {
      if (!effect_functions.contains(it->second)) {
        effect_functions.insert(it->second);
        roots.push_back(it->second);
      }
    } else {
      auto sites = call_sites_per_fn[effect_op];
      for (auto& site : sites) {
        if (!effect_points.contains(site)) {
          effect_points.insert(site);
          roots.push_back(site);
        }
      }
    }
    roots.pop_front();
  }

  // 3. Collect arguments to functions that need to be flattened for no-alias
  //    annotations. In no particular order w.r.t. the other steps.

  root->walk([&](FuncOp op) {
    llvm::SmallVector<size_t, 1> flattened_args;
    for (size_t i = 0; i < op.getNumArguments(); ++i) {
      if (op.getArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName()) &&
          op.getArgument(i).getType().isa<BoundedBufferType>()) {
        flattened_args.push_back(i);
      }
    }
    if (flattened_args.size() > 0) {
      flattened_buffer_args.insert(
          {op.getName().str(), std::move(flattened_args)});
    }
  });
}

Span<size_t> SideEffectAnalysis::flattenedBufferArguments(
    llvm::StringRef callee) const {
  if (auto it = flattened_buffer_args.find(callee);
      it != flattened_buffer_args.end()) {
    return Span<size_t>{&it->second[0], it->second.size()};
  }
  return {};
}

}  // namespace pj
