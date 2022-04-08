#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include <llvm/Target/TargetMachine.h>

#include "defer.hpp"
#include "ir.hpp"
#include "llvm_gen_base.hpp"
#include "side_effect_analysis.hpp"
#include "util.hpp"

namespace pj {
using namespace mlir;
using namespace mlir::LLVM;

using namespace ir2;
using namespace types;

namespace {
struct LLVMGenPass
    : public mlir::PassWrapper<LLVMGenPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  LLVMGenPass(const llvm::TargetMachine* machine) : machine(machine) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ir::ProtoJitDialect, LLVMDialect>();
  }
  void runOnOperation() final;

  std::pair<Value, Value> getEffectDefsFor(Operation* op) {
    auto* provider = effect_analysis->effectProviderFor(op);
    assert(provider && effect_defs.count(provider));
    return effect_defs[provider];
  }

  Width wordSize() const { return Bytes(machine->getPointerSize(0)); };

  mlir::IntegerType wordType() {
    return mlir::IntegerType::get(&getContext(), wordSize().bits(),
                                  mlir::IntegerType::Signless);
  }

  mlir::Type intType(Width width) {
    return mlir::IntegerType::get(&getContext(), width.bits(),
                                  mlir::IntegerType::Signless);
  }

  mlir::Attribute intAttr(Width width, size_t value) {
    return mlir::IntegerAttr::get(intType(width), value);
  }

  mlir::Value buildWordConstant(mlir::Location& loc, mlir::OpBuilder& _,
                                size_t value) {
    return buildIntConstant(loc, _, wordSize(), value);
  }

  mlir::Value buildIntConstant(mlir::Location& loc, mlir::OpBuilder& _,
                               Width width, size_t value) {
    return _.create<mlir::ConstantOp>(loc, intType(width),
                                      intAttr(width, value));
  }

  auto bytePtrType() {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), Bytes(1).bits()));
  }

  auto wordPtrType() {
    return mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(
        &getContext(), wordSize().bits(), mlir::IntegerType::Signless));
  }

  const llvm::TargetMachine* const machine;
  SideEffectAnalysis* effect_analysis;
  llvm::DenseMap<Operation*, std::pair<Value, Value>> effect_defs;
};

// Convert ProtoJit IR types to LLVM.
struct ProtoJitTypeConverter : public mlir::LLVMTypeConverter {
  ProtoJitTypeConverter(mlir::MLIRContext* ctx, LLVMGenPass* pass)
      : mlir::LLVMTypeConverter(ctx), pass(pass) {
    auto bounded_buf_type = mlir::LLVM::LLVMStructType::getNewIdentified(
        &getContext(), "!pj.bbuf", {pass->bytePtrType(), pass->wordType()});

    addConversion(
        [=](pj::types::ValueType type) { return pass->bytePtrType(); });
    addConversion(
        [=](pj::types::RawBufferType type) { return pass->bytePtrType(); });
    addConversion(
        [=](pj::ir::UserStateType type) { return pass->bytePtrType(); });
    addConversion(
        [=](pj::types::BoundedBufferType type) { return bounded_buf_type; });
  }

  LLVMGenPass* pass;
};

}  // namespace

struct ProjectOpLowering : public OpConversionPattern<ProjectOp> {
  ProjectOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                    LLVMGenPass* pass)
      : OpConversionPattern<ProjectOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(ProjectOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  FuncOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                 LLVMGenPass* pass)
      : OpConversionPattern<FuncOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final {
    if (!pass->effect_analysis->hasEffects(op)) {
      return failure();
    }

    auto* ctx = _.getContext();
    _.startRootUpdate(op);

    // success
    op.insertArgument(op.getNumArguments(), pass->wordPtrType(),
                      DictionaryAttr::get(ctx));
    // callback
    op.insertArgument(op.getNumArguments(), pass->wordPtrType(),
                      DictionaryAttr::get(ctx));

    pass->effect_defs[op] = {
        op.getArgument(op.getNumArguments() - 2),
        op.getArgument(op.getNumArguments() - 1),
    };

    _.finalizeRootUpdate(op);
    return success();
  }

  LLVMGenPass* const pass;
};

LogicalResult ProjectOpLowering::matchAndRewrite(
    ProjectOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto source = operands[0];
  auto src_type = op.src().getType();
  auto result = op.getResult().getType();

  if (src_type.isa<types::ValueType>() ||
      src_type.isa<types::RawBufferType>()) {
    // ValueTypes and RawBuffers are represented as 'char*' in LLVM.
    // The output should have the same representation. We can't create
    // a bounded buffer anyway without some reference for the size.
    ASSERT(result.isa<types::ValueType>() ||
           result.isa<types::RawBufferType>());

    auto loc = source.getLoc();
    Value val =
        _.create<GEPOp>(loc, pass->bytePtrType(), source,
                        pass->buildWordConstant(loc, _, op.offset().bytes()));
    _.replaceOp(op, val);
  } else {
    // TODO: handle bounded buffer
    assert(false);
  }

  return success();
}

struct TranscodeOpLowering : public OpConversionPattern<TranscodeOp> {
  TranscodeOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                      LLVMGenPass* pass)
      : OpConversionPattern<TranscodeOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(TranscodeOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult TranscodeOpLowering::matchAndRewrite(
    TranscodeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto src_type = op.src().getType();
  auto dst_type = op.dst().getType();

  // Handle conversion of primitives.
  if (src_type.isa<IntType>() && dst_type.isa<IntType>()) {
    auto src = src_type.cast<IntType>(), dst = dst_type.cast<IntType>();

    if (src->width != dst->width && src->sign != dst->sign) {
      // If signness is different, cannot convert size.
      return failure();
    }

    Value src_ptr = _.create<BitcastOp>(loc, LLVMPointerType::get(src.toMLIR()),
                                        operands[0]);
    Value dst_ptr = _.create<BitcastOp>(loc, LLVMPointerType::get(dst.toMLIR()),
                                        operands[1]);

    Value val = _.create<LoadOp>(loc, src_ptr, src->alignment.bytes());

    // Zero, sign extend, or truncate if necessary.
    if (src->width < dst->width) {
      if (src->sign == Int::Sign::kSigned) {
        val = _.create<SExtOp>(loc, dst.toMLIR(), val);
      } else {
        val = _.create<ZExtOp>(loc, dst.toMLIR(), val);
      }
    } else {
      val = _.create<TruncOp>(loc, dst.toMLIR(), val);
    }

    _.create<StoreOp>(op.getLoc(), val, dst_ptr, dst->alignment.bytes());

    _.eraseOp(op);
    return success();
  }

  // Other forms of transcode are not legal at this point.
  return failure();
}

struct TagOpLowering : public OpConversionPattern<TagOp> {
  TagOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                LLVMGenPass* pass)
      : OpConversionPattern<TagOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(TagOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult TagOpLowering::matchAndRewrite(
    TagOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  auto var_type = op.dst().getType().cast<VariantType>();

  auto tag_ptr = _.create<GEPOp>(
      loc, pass->bytePtrType(), operands[0],
      pass->buildWordConstant(loc, _, var_type.tag_offset().bytes()));

  auto store_ptr = _.create<BitcastOp>(
      loc, LLVMPointerType::get(pass->intType(var_type.tag_width())), tag_ptr);

  auto tag_cst = pass->buildIntConstant(loc, _, var_type.tag_width(), op.tag());

  _.create<StoreOp>(op.getLoc(), tag_cst, store_ptr);

  _.eraseOp(op);
  return success();
}

struct MatchOpLowering : public OpConversionPattern<MatchOp> {
  MatchOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                  LLVMGenPass* pass)
      : OpConversionPattern<MatchOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(MatchOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult MatchOpLowering::matchAndRewrite(
    MatchOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  // Load the tag so we can switch on it.
  auto var_type = op.var().getType().cast<VariantType>();

  auto tag_ptr = _.create<GEPOp>(
      loc, pass->bytePtrType(), operands[0],
      pass->buildWordConstant(loc, _, var_type.tag_offset().bytes()));

  auto load_ptr = _.create<BitcastOp>(
      loc, LLVMPointerType::get(pass->intType(var_type.tag_width())), tag_ptr);

  Value tag_val = _.create<LoadOp>(op.getLoc(), load_ptr);

  // TODO: llvm case expects 32-bit ints, we allow up to 64-bit tags
  tag_val = _.create<ZExtOp>(op.getLoc(), pass->intType(Bits(32)), tag_val);

  llvm::SmallVector<int32_t, 4> case_vals;
  for (auto& term : var_type.terms()) {
    case_vals.emplace_back(term.tag);
  }
  std::sort(case_vals.begin(), case_vals.end());

  // Switch on the tag and dispatch to the appropriate branch.
  _.create<LLVM::SwitchOp>(loc,
                           /*value=*/tag_val,
                           /*defaultDestination=*/op.successors()[0],
                           /*defaultOperands=*/ValueRange{},
                           /*caseValues=*/case_vals,
                           /*caseDestinations=*/op.successors().drop_front(),
                           /*caseOperands=*/ArrayRef<ValueRange>{});

  _.eraseOp(op);
  return success();
}

struct InvokeCallbackOpLowering : public OpConversionPattern<InvokeCallbackOp> {
  InvokeCallbackOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                           LLVMGenPass* pass)
      : OpConversionPattern<InvokeCallbackOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(InvokeCallbackOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult InvokeCallbackOpLowering::matchAndRewrite(
    InvokeCallbackOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  llvm::SmallVector<Type, 2> dispatch_args = {
      operands[0].getType(),
      operands[1].getType(),
  };
  auto void_ty = LLVM::LLVMVoidType::get(_.getContext());
  auto dispatch_fn_type = LLVM::LLVMFunctionType::get(void_ty, dispatch_args);
  auto dispatch_type = LLVM::LLVMPointerType::get(dispatch_fn_type, 0);

  auto [__, callback_store] = pass->getEffectDefsFor(op);

  Value callback = _.create<LLVM::LoadOp>(loc, callback_store);

  // Check if the callback is set
  auto zero = _.create<LLVM::ConstantOp>(
      loc, callback.getType(), _.getIntegerAttr(_.getIntegerType(64), 0));
  auto cond =
      _.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, callback, zero);

  Block* cur_block = _.getBlock();
  Block* end_block = _.splitBlock(cur_block, _.getInsertionPoint());
  Block* call_block = _.createBlock(cur_block->getParent());

  _.setInsertionPointToEnd(cur_block);
  _.create<LLVM::CondBrOp>(loc, cond, end_block, ValueRange{}, call_block,
                           ValueRange{});

  _.setInsertionPointToStart(call_block);
  callback = _.create<LLVM::IntToPtrOp>(loc, dispatch_type, callback);
  auto call = _.create<LLVM::CallOp>(
      loc, TypeRange{LLVM::LLVMVoidType::get(_.getContext())},
      ValueRange{callback, operands[0], operands[1]});
  ASSERT(call.verify().succeeded());
  _.create<LLVM::BrOp>(loc, ValueRange{}, end_block);

  _.eraseOp(op);
  _.setInsertionPointToStart(end_block);
  return success();
}

struct DecodeCatchOpLowering : public OpConversionPattern<DecodeCatchOp> {
  DecodeCatchOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                        LLVMGenPass* pass)
      : OpConversionPattern<DecodeCatchOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(DecodeCatchOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

struct SetCallbackOpLowering : public OpConversionPattern<SetCallbackOp> {
  SetCallbackOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                        LLVMGenPass* pass)
      : OpConversionPattern<SetCallbackOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(SetCallbackOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult SetCallbackOpLowering::matchAndRewrite(
    SetCallbackOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto [__, callback_store] = pass->getEffectDefsFor(op);

  auto target = pass->buildWordConstant(loc, _, op.target().getZExtValue());
  _.create<LLVM::StoreOp>(loc, target, callback_store);

  _.eraseOp(op);
  return success();
}

LogicalResult DecodeCatchOpLowering::matchAndRewrite(
    DecodeCatchOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  // Create real definitions for the anchors and replace anchor uses with
  // them.
  auto one = pass->buildWordConstant(loc, _, 1);
  auto check_alloc = _.create<AllocaOp>(loc, pass->wordPtrType(), one);
  auto callback_alloc = _.create<AllocaOp>(loc, pass->wordPtrType(), one);

  pass->effect_defs[op] = {check_alloc, callback_alloc};

  // Initialize the allocations.
  auto zero = pass->buildWordConstant(loc, _, 0);
  _.create<StoreOp>(loc, zero, check_alloc);
  _.create<StoreOp>(loc, zero, callback_alloc);

  // Replace uses of the op with the YieldOp value.
  auto* yield_op = &op.body().front().back();
  assert(YieldOp::classof(yield_op));

  auto yield_val = YieldOp{yield_op}.result();

  _.replaceOp(op, yield_val);
  _.eraseOp(yield_op);

  // Stitch the body in into the current block.
  // TODO: branch out when we see Check ops.
  auto* start = _.getInsertionBlock();
  auto* body_entry = &op.body().front();
  auto* continuation =
      _.splitBlock(_.getInsertionBlock(), _.getInsertionPoint());
  _.inlineRegionBefore(op.body(), continuation);

  _.setInsertionPointToEnd(start);
  _.create<mlir::BranchOp>(loc, body_entry, ValueRange{});

  _.setInsertionPointToEnd(body_entry);
  _.create<mlir::BranchOp>(loc, continuation, ValueRange{});

  return success();
}

struct CallOpLowering : public OpConversionPattern<mlir::CallOp> {
  CallOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                 LLVMGenPass* pass)
      : OpConversionPattern<mlir::CallOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(mlir::CallOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult CallOpLowering::matchAndRewrite(
    mlir::CallOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  if (!pass->effect_analysis->hasEffects(op)) {
    return failure();
  }

  auto [check, cb] = pass->getEffectDefsFor(op);

  llvm::SmallVector<Value, 4> new_operands;
  new_operands.insert(new_operands.end(), operands.begin(), operands.end());
  new_operands.insert(new_operands.end(), {check, cb});

  auto call = _.create<mlir::CallOp>(op.getLoc(), op.getResultTypes(),
                                     op.callee(), new_operands);
  if (call.getNumResults() > 0) {
    _.replaceOp(op, call.getResults());
  } else {
    _.eraseOp(op);
  }

  return success();
}

void LLVMGenPass::runOnOperation() {
  auto* ctx = &getContext();

  effect_analysis = &getAnalysis<SideEffectAnalysis>();

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVMDialect>();
  target.addLegalOp<ModuleOp>();

  OwningRewritePatternList patterns(ctx);
  ProtoJitTypeConverter type_converter(ctx, this);
  populateStdToLLVMConversionPatterns(type_converter, patterns);
  patterns.add<CallOpLowering, FuncOpLowering, InvokeCallbackOpLowering,
               DecodeCatchOpLowering, ProjectOpLowering, TranscodeOpLowering,
               TagOpLowering, MatchOpLowering, SetCallbackOpLowering>(
      type_converter, ctx, this);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  getOperation().walk([](LLVMFuncOp op) {
    if (op.getName()[0] == '#') {
      op.linkageAttr(LinkageAttr::get(op.getContext(), Linkage::Internal));
    }
  });
}

std::unique_ptr<Pass> createLLVMGenPass(const llvm::TargetMachine* machine) {
  return std::make_unique<LLVMGenPass>(machine);
}

}  // namespace pj
