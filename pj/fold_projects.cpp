#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "ir.hpp"
#include "util.hpp"

namespace pj {
using namespace mlir;
using namespace ir;

struct FoldLayeredProject : public mlir::OpRewritePattern<ProjectOp> {
  FoldLayeredProject(mlir::MLIRContext* context)
      : OpRewritePattern<ProjectOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(ProjectOp op,
                                      mlir::PatternRewriter& _) const override {
    auto offset = op.getOperand(1).getDefiningOp<ConstantOp>();
    if (!offset) return failure();

    auto input = op.getOperand(1).getDefiningOp<ProjectOp>();
    if (!input) return failure();

    auto offset2 = input.getOperand(1).getDefiningOp<ConstantOp>();
    if (!offset2) return failure();

    auto combined =
        offset.value().cast<IntegerAttr>().getValue().getZExtValue() +
        offset2.value().cast<IntegerAttr>().getValue().getZExtValue();

    _.replaceOpWithNewOp<ProjectOp>(op, op.getType(), input.getOperand(0),
                                    combined);

    return success();
  }
};

struct FoldArrayProject : public mlir::OpRewritePattern<XArrayOp> {
  FoldArrayProject(mlir::MLIRContext* context)
      : OpRewritePattern<XArrayOp>(context, /*benefit=*/1) {}

  void TryRewriteBlockArg(mlir::PatternRewriter& _, Block& block,
                          intptr_t i) const {
    if (!block.getArgument(i).hasOneUse()) return;

    auto& use = *block.getArgument(i).use_begin();
    auto* user = use.getOwner();

    if (!isa<ProjectOp>(user)) return;

    auto proj = cast<ProjectOp>(user);

    auto offset = proj.getOperand(1).getDefiningOp<ConstantOp>();
    if (!offset || offset.value().cast<IntegerAttr>().getUInt() != 0) {
      return;
    }

    if (proj.getType().cast<PJType>()->total_size() !=
            block.getArgument(i).getType().cast<PJType>()->total_size() ||
        proj.getType().cast<PJType>()->alignment() !=
            block.getArgument(i).getType().cast<PJType>()->alignment()) {
      // Currently the from-type is used to determine the stride of the
      // conversion loop, so if we change it's size or alignment, the
      // element will be accessed at the wrong address.
      // TODO: set the from and to stride as attributes on XArrayOp instead
      return;
    }

    block.getArgument(i).setType(proj.getType());

    _.replaceOp(proj, block.getArgument(i));

    return;
  }

  mlir::LogicalResult matchAndRewrite(XArrayOp op,
                                      mlir::PatternRewriter& _) const override {
    TryRewriteBlockArg(_, *op.xvalue().begin(), 0);
    TryRewriteBlockArg(_, *op.xvalue().begin(), 1);
    TryRewriteBlockArg(_, *op.xdefault().begin(), 0);
    return success();
  }
};

void ProjectOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList& results, mlir::MLIRContext* context) {
  results.insert<FoldLayeredProject>(context);
  results.insert<FoldArrayProject>(context);
}

}  // namespace pj
