#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "abstract_types.hpp"
#include "arch.hpp"
#include "exceptions.hpp"
#include "ir.hpp"

namespace pj {
using namespace ir;

namespace {
// Convert ProtoJit IR types to LLVM.
class ProtoJitTypeConverter : public mlir::LLVMTypeConverter {
 public:
  ProtoJitTypeConverter(mlir::MLIRContext* C);
};

Type llvmInternalType(MLIRContext* c, const CIntType* type) {
  return IntegerType::get(c, type->abs()->len.bits());
}

static Value getIntConstant(ConversionPatternRewriter& _, Location l,
                            uint64_t i) {
  auto type = IntegerType::get(_.getContext(), 64);
  return _.create<LLVM::ConstantOp>(l, type,
                                    _.getIntegerAttr(_.getIntegerType(64), i));
}

struct XIntOpLowering : public OpConversionPattern<XIntOp> {
  using OpConversionPattern<XIntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(XIntOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto* c = _.getContext();
    const auto* from_type = op.from().getType().cast<PJType>()->AsInt();
    const auto* to_type = op.to().getType().cast<PJType>()->AsInt();

    assert(from_type->alignment().IsBytes() && to_type->alignment().IsBytes());
    assert(from_type->total_size().IsAlignedTo(from_type->alignment()) &&
           to_type->total_size().IsAlignedTo(to_type->alignment()));

    Value value = _.create<LLVM::LoadOp>(op.getLoc(), operands[0],
                                         from_type->alignment().bytes());

    auto to_llvm = llvmInternalType(c, to_type);
    if (from_type->abs()->len < to_type->abs()->len) {
      if (to_type->abs()->conv == AIntType::Conversion::kSigned &&
          from_type->abs()->conv == AIntType::Conversion::kSigned) {
        value = _.create<LLVM::SExtOp>(op.getLoc(), to_llvm, value);
      } else {
        value = _.create<LLVM::ZExtOp>(op.getLoc(), to_llvm, value);
      }
    } else if (from_type->abs()->len > to_type->abs()->len) {
      value = _.create<LLVM::TruncOp>(op.getLoc(), to_llvm, value);
    }

    _.create<LLVM::StoreOp>(op.getLoc(), value, operands[1],
                            to_type->alignment().bytes());
    _.eraseOp(op);
    return success();
  }
};

struct IIntOpLowering : public OpConversionPattern<IIntOp> {
  using OpConversionPattern<IIntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(IIntOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();
    auto type = op.to().getType().cast<PJType>();

    auto llvm_int_type = llvmInternalType(op->getContext(), type->AsInt());
    auto store_val = _.create<LLVM::ConstantOp>(
        l, llvm_int_type,
        _.getIntegerAttr(_.getIntegerType(type->AsInt()->abs()->len.bits()),
                         0));

    _.create<LLVM::StoreOp>(l, store_val, operands[0],
                            type->alignment().bytes());

    _.eraseOp(op);

    return success();
  }
};

static LogicalResult matchAndRewriteRegionalOp(Operation* operation,
                                               ArrayRef<Value> operands,
                                               ConversionPatternRewriter& _) {
  assert(operation->hasTrait<OpTrait::OneRegion>());
  auto l = operation->getLoc();
  Region& body = operation->getRegion(0);

  auto* start = _.getBlock();
  auto* end = _.splitBlock(start, _.getInsertionPoint());
  assert(end->getNumArguments() == 0);

  TypeRange types;
  for (auto& block : body.getBlocks()) {
    auto* term = &block.back();
    if (isa<RetOp>(term)) {
      RetOp ret(term);

      _.setInsertionPointToEnd(&block);
      _.create<BranchOp>(ret.getLoc(), end, ret.getOperands());
      _.eraseOp(ret);

      types = ret.getOperandTypes();
    }
  }

  if (types.size() > 0) {
    end->addArguments(types);
  }

  // Remember which block is the entry before inlining.
  auto* entry = &body.front();

  _.inlineRegionBefore(body, end);

  _.setInsertionPointToEnd(start);
  _.create<BranchOp>(l, entry);

  _.setInsertionPointToStart(end);
  _.eraseOp(operation);

  return success();
}

struct XStrOpLowering : public OpConversionPattern<XStrOp> {
  using OpConversionPattern<XStrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(XStrOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    return matchAndRewriteRegionalOp(op.getOperation(), operands, _);
  }
};

static Value rawGEP(ConversionPatternRewriter& _, Value source, Type target,
                    Value offset) {
  auto l = source.getLoc();
  auto cstar = LLVM::LLVMPointerType::get(IntegerType::get(_.getContext(), 8));
  Value val = _.create<LLVM::BitcastOp>(l, cstar, source);
  val = _.create<LLVM::GEPOp>(l, cstar, val, offset);
  return _.create<LLVM::BitcastOp>(l, target, val);
}

// TODO(3): make architecture-independent
// TODO(7): bitfields
static Value emitProject(ConversionPatternRewriter& _, TypeConverter* converter,
                         Location loc, Value from, Value offset, Type result) {
  return rawGEP(_, from, converter->convertType(result), offset);
}

struct XArrayOpLowering : public OpConversionPattern<XArrayOp> {
  using OpConversionPattern<XArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(XArrayOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    auto from_type = op.xvalue().getArgument(0).getType().cast<PJType>();
    auto to_type = op.xvalue().getArgument(1).getType().cast<PJType>();

    auto from_type_ll = typeConverter->convertType(from_type);
    auto to_type_ll = typeConverter->convertType(to_type);

    // TODO: kByte
    auto from_elsize =
        RoundUp(from_type->total_size(), from_type->alignment()).bytes();
    auto to_elsize =
        RoundUp(to_type->total_size(), to_type->alignment()).bytes();

    auto from = operands[0];
    auto to = operands[1];

    if (from_type.cast<PJType>()->IsInt() && to_type.cast<PJType>()->IsInt()) {
      // Corresponding array type is int*, but the LLVM type corresponding
      // to the PJ int type itself is also int*, so we don't need to wrap
      // in a pointer here (that would in fact be wrong to do).
      from = _.create<LLVM::BitcastOp>(l, from_type_ll, from);
      to = _.create<LLVM::BitcastOp>(l, to_type_ll, to);
    }

    auto from_len =
        op->getOperand(0).getType().cast<PJType>()->AsArray()->abs()->length;
    auto to_len =
        op->getOperand(1).getType().cast<PJType>()->AsArray()->abs()->length;

    _.eraseOp(op);

    auto* start = _.getBlock();
    auto* end = _.splitBlock(start, _.getInsertionPoint());

    auto index_type = IntegerType::get(op.getContext(), 64);

    if (to_len == 0) {
      return success();
    }

    // TODO: there's a stupid amount of code duplication here

    if (from_len > 0) {
      // Generate a loop to transcode available elements from the
      // source array.
      auto* join = _.createBlock(end, {index_type});

      _.setInsertionPointToEnd(start);

      // Initial index is 0.
      auto initial = _.create<LLVM::ConstantOp>(
          l, index_type, _.getIntegerAttr(_.getIntegerType(64), 0));

      _.create<LLVM::BrOp>(l, ValueRange{initial}, join);

      _.setInsertionPointToEnd(join);

      auto limit = _.create<LLVM::ConstantOp>(
          l, index_type,
          _.getIntegerAttr(_.getIntegerType(64), std::min(from_len, to_len)));
      auto cond =
          _.create<LLVM::ICmpOp>(l, LLVM::ICmpPredicate::ult,
                                 /*current index*/ join->getArgument(0), limit);

      Block* body = _.createBlock(end, {index_type});

      _.setInsertionPointToEnd(join);
      _.create<LLVM::CondBrOp>(l, cond, body, ValueRange{join->getArgument(0)},
                               end, ValueRange{});

      _.setInsertionPointToEnd(body);
      auto body_index = body->getArgument(0);

      auto from_elsize_v = getIntConstant(_, l, from_elsize);
      auto to_elsize_v = getIntConstant(_, l, to_elsize);
      auto from_offset = _.create<LLVM::MulOp>(l, from_elsize_v, body_index);
      auto to_offset = _.create<LLVM::MulOp>(l, to_elsize_v, body_index);

      auto from_el =
          emitProject(_, typeConverter, l, from, from_offset, from_type);
      auto to_el = emitProject(_, typeConverter, l, to, to_offset, to_type);

      Block* entry = &op.xvalue().front();

      _.create<LLVM::BrOp>(l, ValueRange{from_el, to_el}, entry);
      _.inlineRegionBefore(op.xvalue(), end);

      Block* loop_end = _.createBlock(end);
      auto next_index =
          _.create<LLVM::AddOp>(l, body_index, getIntConstant(_, l, 1));
      _.create<LLVM::BrOp>(l, ValueRange{next_index}, join);

      ReplaceTerminators(_, loop_end, Region::iterator(entry),
                         Region::iterator(loop_end));

      start = end;
      end = _.splitBlock(start, start->begin());
    }

    if (to_len > from_len) {
      // Generate a loop to transcode available elements from the
      // source array.
      auto* join = _.createBlock(end, {index_type});

      _.setInsertionPointToEnd(start);

      // Initial index is 0.
      auto initial = _.create<LLVM::ConstantOp>(
          l, index_type, _.getIntegerAttr(_.getIntegerType(64), from_len));

      _.create<LLVM::BrOp>(l, ValueRange{initial}, join);

      _.setInsertionPointToEnd(join);

      auto limit = _.create<LLVM::ConstantOp>(
          l, index_type, _.getIntegerAttr(_.getIntegerType(64), to_len));
      auto cond =
          _.create<LLVM::ICmpOp>(l, LLVM::ICmpPredicate::ult,
                                 /*current index*/ join->getArgument(0), limit);

      Block* body = _.createBlock(end, {index_type});

      _.setInsertionPointToEnd(join);
      _.create<LLVM::CondBrOp>(l, cond, body, ValueRange{join->getArgument(0)},
                               end, ValueRange{});

      _.setInsertionPointToEnd(body);
      auto body_index = body->getArgument(0);

      auto to_elsize_v = getIntConstant(_, l, to_elsize);
      auto to_offset = _.create<LLVM::MulOp>(l, to_elsize_v, body_index);
      auto to_el = emitProject(_, typeConverter, l, to, to_offset, to_type);

      Block* entry = &op.xdefault().front();

      _.create<LLVM::BrOp>(l, ValueRange{to_el}, entry);
      _.inlineRegionBefore(op.xdefault(), end);

      Block* loop_end = _.createBlock(end);
      auto next_index =
          _.create<LLVM::AddOp>(l, body_index, getIntConstant(_, l, 1));
      _.create<LLVM::BrOp>(l, ValueRange{next_index}, join);

      ReplaceTerminators(_, loop_end, Region::iterator(entry),
                         Region::iterator(loop_end));
    } else {
      _.setInsertionPointToEnd(start);
      _.create<LLVM::BrOp>(l, ValueRange{}, end);
    }

    return success();
  }
};

struct ProjectOpLowering : public OpConversionPattern<ProjectOp> {
  using OpConversionPattern<ProjectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ProjectOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    _.replaceOp(op, emitProject(_, typeConverter, op.getLoc(), operands[0],
                                operands[1], op.getType()));

    return success();
  }
};

// TODO(28): reduce redundancy between ETagOp and DTagOp

struct ETagOpLowering : public OpConversionPattern<ETagOp> {
  using OpConversionPattern<ETagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ETagOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    assert(op.offset().IsBytes());

    auto tag_width = op.tag().getType().cast<IntegerType>().getWidth();
    auto llvm_int_type = IntegerType::get(op.getContext(), tag_width);

    // TODO(25): allow higher alignment
    // TODO(7): bitfields -- allow lower alignment
    auto store_ptr_type = LLVM::LLVMPointerType::get(llvm_int_type, 0);
    auto into = operands[0];
    {
      auto addr_type = IntegerType::get(op.getContext(), 64);
      into = _.create<LLVM::PtrToIntOp>(l, addr_type, into);
      auto offset_val = _.create<LLVM::ConstantOp>(
          l, addr_type,
          _.getIntegerAttr(_.getIntegerType(64), op.offset().bytes()));
      into = _.create<LLVM::AddOp>(l, into, offset_val);
      into = _.create<LLVM::IntToPtrOp>(l, store_ptr_type, into);
    }

    _.create<LLVM::StoreOp>(l, operands[1], into, /*alignment=*/1);
    _.eraseOp(op);
    return success();
  }
};

struct DTagOpLowering : public OpConversionPattern<DTagOp> {
  using OpConversionPattern<DTagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(DTagOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    auto tag_width = op.getType().cast<IntegerType>().getWidth();
    auto llvm_int_type = IntegerType::get(op.getContext(), tag_width);

    // TODO(25): allow higher alignment
    // TODO(7): bitfields -- allow lower alignment
    auto load_ptr_type = LLVM::LLVMPointerType::get(llvm_int_type, 0);
    auto from = operands[0];
    {
      auto addr_type = IntegerType::get(op.getContext(), 64);
      from = _.create<LLVM::PtrToIntOp>(l, addr_type, from);
      auto offset_val = _.create<LLVM::ConstantOp>(
          l, addr_type,
          _.getIntegerAttr(_.getIntegerType(64), op.offset().bytes()));
      from = _.create<LLVM::AddOp>(l, from, offset_val);
      from = _.create<LLVM::IntToPtrOp>(l, load_ptr_type, from);
    }

    Value tag = _.create<LLVM::LoadOp>(l, llvm_int_type, from, /*alignment=*/1);
    _.replaceOp(op, tag);
    return success();
  }
};

struct LTagOpLowering : public OpConversionPattern<LTagOp> {
  using OpConversionPattern<LTagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(LTagOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    auto result_tag_width =
        op.getType().isa<IndexType>()
            ? Bits(64)
            : Bits(op.getType().cast<IntegerType>().getWidth());

    auto result_int_type =
        IntegerType::get(op.getContext(), result_tag_width.bits());
    auto load_tag_width = op.width();
    auto load_int_type =
        IntegerType::get(op.getContext(), load_tag_width.bits());

    // TODO(25): allow higher alignment
    // TODO(7): bitfields -- allow lower alignment
    auto load_ptr_type = LLVM::LLVMPointerType::get(load_int_type, 0);
    auto from = operands[0];
    {
      auto addr_type = IntegerType::get(op.getContext(), 64);
      from = _.create<LLVM::PtrToIntOp>(l, addr_type, from);
      from = _.create<LLVM::AddOp>(l, from, operands[1]);
      from = _.create<LLVM::IntToPtrOp>(l, load_ptr_type, from);
    }

    Value tag = _.create<LLVM::LoadOp>(l, load_int_type, from, /*alignment=*/1);
    if (result_tag_width != load_tag_width) {
      // Zero-extend
      tag = _.create<LLVM::ZExtOp>(l, result_int_type, tag);
    }
    _.replaceOp(op, tag);
    return success();
  }
};

struct LRefOpLowering : public OpConversionPattern<LRefOp> {
  using OpConversionPattern<LRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(LRefOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    auto base = operands[0];
    auto ref_offset = op.ref_offset();
    auto ref_size = op.ref_size();

    auto cstar =
        LLVM::LLVMPointerType::get(IntegerType::get(_.getContext(), 8));
    auto raw_base = _.create<LLVM::BitcastOp>(l, cstar, base);

    auto llvm_ref_type = IntegerType::get(op.getContext(), ref_size.bits());
    auto load_ref_type = LLVM::LLVMPointerType::get(llvm_ref_type, 0);
    auto ref_base0 = _.create<LLVM::GEPOp>(
        l, cstar, raw_base, getIntConstant(_, l, ref_offset.bytes()));
    auto ref_base = _.create<LLVM::BitcastOp>(l, load_ref_type, ref_base0);
    // TODO: alignment could be more precise.
    auto ref = _.create<LLVM::LoadOp>(l, ref_base, 1);

    auto raw_result =
        _.create<LLVM::GEPOp>(l, cstar, raw_base, ValueRange{ref});
    auto result = _.create<LLVM::BitcastOp>(
        l, typeConverter->convertType(op.getType().cast<PJType>()), raw_result);

    _.replaceOp(op, {result});
    return success();
  }
};

struct BTagOpLowering : public OpConversionPattern<BTagOp> {
  using OpConversionPattern<BTagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BTagOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();
    auto tag = operands[0];

    // TODO(26): how should we treat a missing variant case?

    // Unfortunately the LLVM Switch instruction is not yet available through
    // the LLVM MLIR dialect, so we generate a chain of CondBrOps instead.
    // LLVM will usually apply a dispatch table optimization anyway, but we
    // really should ensure it. Also, we could build a more efficient dispatch
    // table than LLVM with our own backend, leveraging the fact that the
    // branch targets are very uniform in size.
    assert(op.getNumSuccessors() > 0);

    if (op.getNumSuccessors() == 1) {
      _.create<LLVM::BrOp>(l, ValueRange{}, op.getSuccessor(0));
      _.eraseOp(op);
      return success();
    }

    auto* start = _.getBlock();
    _.eraseOp(op);
    _.setInsertionPointToEnd(start);

    for (intptr_t i = 0; i < op.getNumSuccessors() - 1; ++i) {
      const auto match = op.tagOptions().getValue(i);
      auto match_val = _.create<LLVM::ConstantOp>(l, tag.getType(), match);
      auto cmp =
          _.create<LLVM::ICmpOp>(l, LLVM::ICmpPredicate::eq, match_val, tag);
      Block* fail_block = nullptr;
      if (i == op.getNumSuccessors() - 2) {
        fail_block = op.getSuccessor(i + 1);
      } else {
        fail_block = _.createBlock(op.getSuccessor(i), TypeRange{});
      }
      _.setInsertionPointToEnd(start);
      _.create<LLVM::CondBrOp>(l, cmp, op.getSuccessor(i), fail_block);
      _.setInsertionPointToStart(fail_block);
      start = fail_block;
    }

    return success();
  }
};

struct DispatchOpLowering : public OpConversionPattern<DispatchOp> {
  using OpConversionPattern<DispatchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(DispatchOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    llvm::SmallVector<Type, 1> dispatch_args = {operands[0].getType(),
                                                operands[1].getType()};
    auto void_ty = LLVM::LLVMVoidType::get(_.getContext());
    auto dispatch_fn_type = LLVM::LLVMFunctionType::get(void_ty, dispatch_args);
    auto dispatch_type = LLVM::LLVMPointerType::get(dispatch_fn_type, 0);

    // TODO: architecture-dependent
    auto llvm_int_type = IntegerType::get(op.getContext(), 64);
    Value target_val = _.create<LLVM::ConstantOp>(
        l, llvm_int_type,
        _.getIntegerAttr(_.getIntegerType(64, /*signed=*/false), op.target()));
    target_val = _.create<LLVM::IntToPtrOp>(l, dispatch_type, target_val);

    _.create<LLVM::CallOp>(l, TypeRange{void_ty},
                           ValueRange{target_val, operands[0], operands[1]});
    _.eraseOp(op);

    return success();
  }
};

struct IterOpLowering : public OpConversionPattern<IterOp> {
  using OpConversionPattern<IterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(IterOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    auto l = op.getLoc();

    auto* start = _.getBlock();
    auto* end = _.splitBlock(start, _.getInsertionPoint());

    auto initial_index = operands[0];
    auto limit_index = operands[1];
    ArrayRef<Value> initial_ivs{operands.begin() + 2, operands.end()};

    std::vector<Value> ops;

    Block *join = nullptr, *pre_join = nullptr;
    {
      ops.push_back(initial_index);
      for (auto operand : initial_ivs) {
        ops.push_back(operand);
      }

      std::vector<Type> join_types;
      for (auto val : ops) {
        join_types.push_back(val.getType());
      }

      join = _.createBlock(end, join_types);

      ArrayRef<Type> pre_join_types{join_types.data() + 1,
                                    join_types.size() - 1};

      pre_join = _.createBlock(end, pre_join_types);

      for (auto& type : pre_join_types) {
        end->addArgument(type);
      }

      _.setInsertionPointToEnd(start);
      _.create<BranchOp>(l, join, ops);

      ops.clear();
    }

    auto* body_entry = &op.body().front();
    _.inlineRegionBefore(op.body(), end);

    ReplaceTerminators(_, pre_join, Region::iterator{body_entry},
                       Region::iterator{end}, /*update_join_args=*/false);

    {
      _.setInsertionPointToStart(pre_join);
      auto next_index = _.create<LLVM::AddOp>(l, join->getArgument(0),
                                              getIntConstant(_, l, 1));
      ops.push_back(next_index);
      for (intptr_t i = 0; i < pre_join->getNumArguments(); ++i) {
        ops.push_back(pre_join->getArgument(i));
      }
      _.create<BranchOp>(l, join, ops);

      ops.clear();
    }

    {
      _.setInsertionPointToEnd(join);
      auto cond = _.create<LLVM::ICmpOp>(l, LLVM::ICmpPredicate::ult,
                                         join->getArgument(0), limit_index);

      for (intptr_t i = 0; i < join->getNumArguments(); ++i) {
        ops.push_back(join->getArgument(i));
      }
      auto end_args = ArrayRef<Value>{ops.data() + 1, ops.size() - 1};
      _.create<LLVM::CondBrOp>(l, cond, body_entry, ops, end, end_args);

      ops.clear();
    }

    _.setInsertionPointToStart(end);

    {
      for (intptr_t i = 0; i < end->getNumArguments(); ++i) {
        ops.push_back(end->getArgument(i));
      }

      _.replaceOp(op, ops);

      ops.clear();
    }

    return success();
  }
};

namespace {
struct ProtoJitToLLVMLoweringPass
    : public PassWrapper<ProtoJitToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
}  // namespace

void ProtoJitToLLVMLoweringPass::runOnOperation() {
  auto* c = &getContext();

  ConversionTarget target(*c);
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<ir::CastOp>();

  ProtoJitTypeConverter type_converter(c);

  OwningRewritePatternList patterns(c);
  populateStdToLLVMConversionPatterns(type_converter, patterns);
  patterns.insert<         //
      BTagOpLowering,      //
      IIntOpLowering,      //
      DTagOpLowering,      //
      LTagOpLowering,      //
      LRefOpLowering,      //
      ETagOpLowering,      //
      ProjectOpLowering,   //
      XIntOpLowering,      //
      XStrOpLowering,      //
      DispatchOpLowering,  //
      XArrayOpLowering,    //
      IterOpLowering       //
      >(type_converter, c);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

static Optional<Value> sourceIndexMaterialization(OpBuilder& _,
                                                  mlir::IndexType type,
                                                  ValueRange inputs,
                                                  Location loc) {
  if (inputs.size() != 1) {
    return None;
  }

  return {_.create<ir::CastOp>(loc, type, inputs[0])};
}

static Optional<Value> targetIndexMaterialization(OpBuilder& _,
                                                  IntegerType type,
                                                  ValueRange inputs,
                                                  Location loc) {
  if (inputs.size() != 1) {
    return None;
  }

  if (!inputs[0].getType().isa<mlir::IndexType>()) {
    return None;
  }

  return {_.create<ir::CastOp>(loc, type, inputs[0])};
}

ProtoJitTypeConverter::ProtoJitTypeConverter(MLIRContext* c)
    : LLVMTypeConverter(c) {
  addConversion([](PJType type) { return type.toLLVM(); });
  addConversion([](UserStateType type) { return type.toLLVM(); });
  addSourceMaterialization(sourceIndexMaterialization);
  addTargetMaterialization(targetIndexMaterialization);
}

}  // namespace

Type PJType::toLLVM() const {
  if ((*this)->IsInt()) {
    return LLVM::LLVMPointerType::get(
        llvmInternalType(getContext(), (*this)->AsInt()), 0);
  }
  auto internal = LLVM::LLVMStructType::getLiteral(getContext(), {});
  return LLVM::LLVMPointerType::get(internal, 0);
}

Type UserStateType::toLLVM() const {
  auto internal = LLVM::LLVMStructType::getLiteral(getContext(), {});
  return LLVM::LLVMPointerType::get(internal, 0);
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ProtoJitToLLVMLoweringPass>();
}

}  // namespace pj
