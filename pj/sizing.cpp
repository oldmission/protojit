#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "ir.hpp"
#include "passes.hpp"

namespace pj {
using namespace mlir;
using namespace ir;
using namespace types;

namespace {
struct GenSizeFunctionsPass
    : public PassWrapper<GenSizeFunctionsPass, OperationPass<ModuleOp>> {
  GenSizeFunctionsPass() {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ir::ProtoJitDialect>();
  }
  void runOnOperation() final;

 private:
  mlir::ModuleOp module() { return mlir::ModuleOp(getOperation()); }

  FuncOp findOrCreateConvertedFn(llvm::StringRef name, bool round_up);

  // Removes all operations not relevant for sizing and rounds up length
  // calculations when possible if round_up is set.
  void convertRegion(Region& region, bool round_up);

  llvm::StringMap<FuncOp> converted_fns_;
};

FuncOp GenSizeFunctionsPass::findOrCreateConvertedFn(llvm::StringRef name,
                                                     bool round_up) {
  std::string conv_name = name.str() + (round_up ? "_rounded" : "_exact");

  auto it = converted_fns_.find(conv_name);
  if (it != converted_fns_.end()) {
    return it->second;
  }

  auto orig = module().lookupSymbol<FuncOp>(name);
  assert(orig);

  auto func = orig.clone();
  func.sym_nameAttr(StringAttr::get(&getContext(), conv_name));

  auto _ = mlir::OpBuilder::atBlockBegin(&*func.begin());
  auto dst = _.create<UnitOp>(_.getUnknownLoc(), orig.getArgument(1).getType());
  func.getArgument(1).replaceAllUsesWith(dst);
  func.eraseArgument(1);

  _.setInsertionPointToStart(module().getBody());
  _.insert(func);

  convertRegion(func.body(), round_up);

  converted_fns_.try_emplace(conv_name, func);
  orig.erase();
  return func;
}

void GenSizeFunctionsPass::convertRegion(Region& region, bool round_up) {
  region.walk([&](Operation* op) {
    if (auto call = dyn_cast<CallOp>(op)) {
      auto loc = call.getLoc();
      auto func = findOrCreateConvertedFn(call.callee(), round_up);

      auto _ = mlir::OpBuilder{call};
      auto new_call = _.create<mlir::CallOp>(
          loc, func, ValueRange{call.operands()[0], call.operands()[2]});

      call.replaceAllUsesWith(ValueRange{new_call.getResult(0)});
      call.erase();
      return;
    }

    if (round_up) {
      // Round up vector lengths that have a maximum size to constants. This
      // does extend loops potentially past the range of the vector, but since
      // the elements also have a maximum size, their AllocateOps will also have
      // been switched to use constant lengths, avoiding invalid reads.
      if (auto len = dyn_cast<LengthOp>(op)) {
        auto vec = len.vec().getType().cast<VectorType>();
        if (vec.hasMaxSize()) {
          auto _ = mlir::OpBuilder{len};
          auto max = _.create<ConstantOp>(
              len.getLoc(),
              _.getIntegerAttr(_.getIndexType(), vec->max_length));

          len.replaceAllUsesWith(max.getResult());
          len.erase();
        }
      }
    }
  });

  // Use a mark and sweep algorithm to remove any instructions that do not
  // directly feed into any of the final buffer return statements. The intention
  // is to keep only AlignOps, AllocateOps and the control flow that determines
  // which of them are executed.
  llvm::SmallVector<Operation*, 8> worklist;
  llvm::SmallSet<Operation*, 16> marked;
  region.walk([&](Operation* op) {
    if (isa<ReturnOp>(op) || isa<MatchOp>(op) || isa<YieldOp>(op) ||
        isa<scf::YieldOp>(op)) {
      worklist.emplace_back(op);
    }
  });

  while (!worklist.empty()) {
    auto op = worklist.pop_back_val();
    marked.insert(op);

    for (auto operand : op->getOperands()) {
      if (auto defining_op = operand.getDefiningOp()) {
        worklist.emplace_back(defining_op);
      }
    }
  }

  region.walk([&](Operation* op) {
    if (!marked.contains(op)) {
      op->dropAllDefinedValueUses();
      op->erase();
    }
  });
}

void GenSizeFunctionsPass::runOnOperation() {
  module().walk([&](SizeOp op) { convertRegion(op.body(), op.round_up()); });
}

}  // namespace

std::unique_ptr<Pass> createGenSizeFunctionsPass() {
  return std::make_unique<GenSizeFunctionsPass>();
}

}  // namespace pj
