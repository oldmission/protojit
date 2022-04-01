#include "inline_regions.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "defer.hpp"
#include "ir.hpp"
#include "util.hpp"

namespace pj {
using namespace mlir;
using namespace ir;

struct XArrayOpLowering : public OpConversionPattern<XArrayOp> {
  using OpConversionPattern<XArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(XArrayOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult XArrayOpLowering::matchAndRewrite(
    XArrayOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto L = op.getLoc();
  auto C = _.getContext();

  auto from = op->getOperand(0);
  auto to = op->getOperand(1);

  const CArrayType* from_type = from.getType().cast<PJType>()->AsArray();
  const CArrayType* to_type = to.getType().cast<PJType>()->AsArray();

  const auto from_elsize =
      RoundUp(from_type->el()->total_size(), from_type->el()->alignment())
          .bytes();
  const auto to_elsize =
      RoundUp(to_type->el()->total_size(), to_type->el()->alignment()).bytes();

  const auto convert_len =
      std::min(from_type->abs()->length, to_type->abs()->length);

  if (convert_len > 0) {
    auto iter = _.create<IterOp>(L,
                                 /*results=*/mlir::TypeRange{},
                                 /*start=*/GetIndexConstant(L, _, 0),
                                 /*end=*/GetIndexConstant(L, _, convert_len),
                                 /*induction_variables=*/mlir::ValueRange{});
    DEFER(_.setInsertionPointAfter(iter));

    auto end = new Block();
    iter.body().push_back(end);
    _.inlineRegionBefore(op.xvalue(), end);

    auto body_start = &iter.body().front();
    auto entry = new Block();
    iter.body().push_front(entry);

    auto index = entry->addArgument(mlir::IndexType::get(C));
    _.setInsertionPointToStart(entry);

    mlir::Value from_offset =
        _.create<MulIOp>(L, GetIndexConstant(L, _, from_elsize), index);
    auto from_el = _.create<ProjectOp>(L, body_start->getArgument(0).getType(),
                                       from, from_offset);

    mlir::Value to_offset =
        _.create<MulIOp>(L, GetIndexConstant(L, _, to_elsize), index);
    auto to_el = _.create<ProjectOp>(L, body_start->getArgument(1).getType(),
                                     to, to_offset);

    _.create<BranchOp>(L, ValueRange{from_el, to_el}, body_start);

    end->erase();
  }

  const auto fill_len =
      std::max(to_type->abs()->length - from_type->abs()->length, 0L);
  if (fill_len > 0) {
    auto iter =
        _.create<IterOp>(L,
                         /*results=*/mlir::TypeRange{},
                         /*start=*/GetIndexConstant(L, _, convert_len),
                         /*end=*/GetIndexConstant(L, _, convert_len + fill_len),
                         /*induction_variables=*/mlir::ValueRange{});
    DEFER(_.setInsertionPointAfter(iter));

    auto end = new Block();
    iter.body().push_back(end);
    _.inlineRegionBefore(op.xdefault(), end);

    auto body_start = &iter.body().front();
    auto entry = new Block();
    iter.body().push_front(entry);

    auto index = entry->addArgument(mlir::IndexType::get(C));
    _.setInsertionPointToStart(entry);

    mlir::Value to_offset =
        _.create<MulIOp>(L, GetIndexConstant(L, _, to_elsize), index);
    auto to_el = _.create<ProjectOp>(L, body_start->getArgument(0).getType(),
                                     to, to_offset);

    _.create<BranchOp>(L, ValueRange{to_el}, body_start);

    end->erase();
  }

  _.setInsertionPointAfter(op);
  _.eraseOp(op);

  return success();
}

struct SListOpLowering : public OpConversionPattern<SListOp> {
  using OpConversionPattern<SListOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SListOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto L = op->getLoc();
    auto C = _.getContext();

    auto source_type = operands[0].getType().cast<PJType>()->AsList();
    auto target_type = op.target().cast<PJType>()->AsList();

    const auto source_elsize = source_type->el->aligned_size();
    const auto target_elsize = target_type->el->aligned_size();

    Value total_external_size = GetIndexConstant(L, _, 0);

    // The external size has two terms:
    // - Size of the (external) array holding elements +
    // - External size of serialized elements in array
    // Both terms depend on the number of serialized elements (N).

    // N:
    Value N;
    {
      // N = min(number of elements in source list,
      //         max elements in target list)

      auto source_len = source_type->LoadLength(L, operands[0], _);
      auto max_target_len = GetIndexConstant(L, _, target_type->abs()->max_len);

      auto is_clamped =
          _.create<CmpIOp>(L, CmpIPredicate::ule, source_len, max_target_len);
      N = _.create<SelectOp>(L, is_clamped, source_len, max_target_len);
    }

    // First term: size of the (external) array
    {
      // ArraySize = (number of external elements) * (target size of element)
      //
      // The first factor depends on whether the target count overflows.
      //
      // Overflow = N > target_full_payload_count
      // OverflowCount = N - target_partial_payload_count
      // ExternalN = Overflow ? OverflowCount : 0
      //
      // Finally:
      //
      // ArraySize = ExternalN * target_element_size

      auto overflow = _.create<CmpIOp>(
          L, CmpIPredicate::ugt, N,
          GetIndexConstant(L, _, target_type->full_payload_count));

      auto overflow_count = _.create<SubIOp>(
          L, N, GetIndexConstant(L, _, target_type->partial_payload_count));

      auto external_n = _.create<SelectOp>(L, overflow, overflow_count,
                                           GetIndexConstant(L, _, 0));

      auto array_size = _.create<MulIOp>(
          L, external_n, GetIndexConstant(L, _, target_elsize.bytes()));
      total_external_size =
          _.create<AddIOp>(L, total_external_size, array_size);
    }

    // Second term: sum of external size of serialized elements
    {
      // Scan through first N elements of the source and invoke the sizing
      // routine on them.
      const auto source_eltype = source_type->el->toIR(C);

      // Right now we assume the source is always outline data, which is true
      // of ArrayView.
      assert(source_type->full_payload_count == 0 &&
             source_type->partial_payload_count == 0);

      auto source_array =
          source_type->LoadOutlinedArray(L, operands[0], source_eltype, _);

      auto iter =
          _.create<IterOp>(L, GetIndexType(_), GetIndexConstant(L, _, 0), N,
                           ValueRange{total_external_size});

      DEFER(_.setInsertionPoint(_.getBlock(), _.getInsertionPoint()));

      auto* header = _.createBlock(&iter.body());

      auto index = header->addArgument(GetIndexType(_));
      auto running_size = header->addArgument(GetIndexType(_));

      auto join =
          _.createBlock(&iter.body(), iter.body().end(), {GetIndexType(_)});

      auto body_start = &op.body().front();
      _.inlineRegionBefore(op.body(), join);
      ReplaceTerminators(_, join, ++Region::iterator(header),
                         Region::iterator(join),
                         /*update_join_args=*/false);

      {
        _.setInsertionPointToStart(header);
        Value source_el_offset = _.create<MulIOp>(
            L, index, GetIndexConstant(L, _, source_elsize.bytes()));
        auto source_el = _.create<ProjectOp>(L, source_eltype, source_array,
                                             source_el_offset);
        _.create<BranchOp>(L, body_start, ValueRange{source_el});
      }

      {
        _.setInsertionPointToStart(join);
        auto updated_size =
            _.create<AddIOp>(L, join->getArgument(0), running_size);
        _.create<RetOp>(L, ValueRange{updated_size});
      }

      total_external_size = iter->getResult(0);
    }

    _.replaceOp(op, total_external_size);
    return success();
  }
};

static LogicalResult matchAndRewriteRegionalOp(Operation* operation,
                                               ArrayRef<Value> operands,
                                               ConversionPatternRewriter& _) {
  assert(operation->hasTrait<OpTrait::OneRegion>());
  auto L = operation->getLoc();
  Region& body = operation->getRegion(0);

  auto start = _.getBlock();
  auto end = _.splitBlock(start, _.getInsertionPoint());
  assert(end->getNumArguments() == 0);

  auto types = ReplaceTerminators(_, end, body.begin(), body.end());

  // Remember which block is the entry before inlining.
  auto entry = &body.front();

  _.inlineRegionBefore(body, end);

  _.setInsertionPointToEnd(start);
  _.create<BranchOp>(L, entry);

  if (types.size() > 0) {
    _.replaceOp(operation, end->getArgument(0));
  } else {
    _.eraseOp(operation);
  }

  _.setInsertionPointToStart(end);

  return success();
}

struct XStrOpLowering : public OpConversionPattern<XStrOp> {
  using OpConversionPattern<XStrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(XStrOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    return matchAndRewriteRegionalOp(op, operands, _);
  }
};

struct MatchVariantOpLowering : public OpConversionPattern<MatchVariantOp> {
  using OpConversionPattern<MatchVariantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatchVariantOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    return matchAndRewriteRegionalOp(op, operands, _);
  }
};

struct SStrOpLowering : public OpConversionPattern<SStrOp> {
  using OpConversionPattern<SStrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SStrOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    return matchAndRewriteRegionalOp(op, operands, _);
  }
};

namespace {
struct InlineRegionsPass
    : public PassWrapper<InlineRegionsPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ir::ProtoJitDialect>();
  }
  void runOnOperation() final;
};
}  // namespace

void InlineRegionsPass::runOnOperation() {
  auto* C = &getContext();

  ConversionTarget target(*C);
  target.addLegalDialect<ir::ProtoJitDialect, StandardOpsDialect>();
  target.addLegalOp<ModuleOp, FuncOp>();
  target.addIllegalOp<ir::XStrOp, ir::MatchVariantOp, ir::SStrOp, ir::XArrayOp,
                      ir::SListOp>();

  OwningRewritePatternList patterns(C);
  patterns.insert<XStrOpLowering, MatchVariantOpLowering, SStrOpLowering,
                  XArrayOpLowering, SListOpLowering>(C);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createInlineRegionsPass() {
  return std::make_unique<InlineRegionsPass>();
}

}  // namespace pj
