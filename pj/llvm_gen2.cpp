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
using namespace ir;
using namespace types;

namespace {
struct LLVMGenPass
    : public mlir::PassWrapper<LLVMGenPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ir::ProtoJitDialect, LLVMDialect>();
  }
  void runOnOperation() final;

  void setEffectDefs(Operation* op, Value check, Value callback) {
    effect_defs_[op] = {check, callback};
  }

  std::pair<Value, Value> getEffectDefsFor(Operation* op) {
    auto* provider = effect_analysis_->effectProviderFor(op);
    assert(provider && effect_defs_.count(provider));
    return effect_defs_[provider];
  }

  Width wordSize() const { return Bytes(machine_->getPointerSize(0)); };

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

  mlir::Type boundedBufType() {
    if (!bounded_buf_type_) {
      bounded_buf_type_ = mlir::LLVM::LLVMStructType::getNewIdentified(
          &getContext(), "!pj.bbuf", {bytePtrType(), wordType()});
    }
    return bounded_buf_type_;
  }

  std::pair<Value, Value> buildBoundedBufferDestructuring(
      Location loc, ConversionPatternRewriter& _, Value buf);

  LLVMGenPass(const llvm::TargetMachine* machine) : machine_(machine) {}

  SideEffectAnalysis* effectAnalysis() { return effect_analysis_; };

 private:
  const llvm::TargetMachine* const machine_;
  mlir::Type bounded_buf_type_;
  SideEffectAnalysis* effect_analysis_;
  llvm::DenseMap<Operation*, std::pair<Value, Value>> effect_defs_;
};

// Convert ProtoJit IR types to LLVM.
struct ProtoJitTypeConverter : public mlir::LLVMTypeConverter {
  ProtoJitTypeConverter(mlir::MLIRContext* ctx, LLVMGenPass* pass)
      : mlir::LLVMTypeConverter(ctx), pass(pass) {
    addConversion([=](ValueType type) { return pass->bytePtrType(); });
    addConversion([=](RawBufferType type) { return pass->bytePtrType(); });
    addConversion([=](UserStateType type) { return pass->bytePtrType(); });
    addConversion(
        [=](BoundedBufferType type) { return pass->boundedBufType(); });
  }

  LLVMGenPass* pass;
};

std::pair<Value, Value> LLVMGenPass::buildBoundedBufferDestructuring(
    Location loc, ConversionPatternRewriter& _, Value buf) {
  return {
      _.create<LLVM::ExtractValueOp>(  //
          loc, bytePtrType(), buf, _.getI64ArrayAttr(0)),
      _.create<LLVM::ExtractValueOp>(  //
          loc, wordType(), buf, _.getI64ArrayAttr(1)),
  };
}

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
    auto* ctx = _.getContext();
    auto loc = op.getLoc();

    _.startRootUpdate(op);

    if (pass->effectAnalysis()->hasEffects(op)) {
      // success
      op.insertArgument(op.getNumArguments(), pass->wordPtrType(),
                        DictionaryAttr::get(ctx));
      // callback
      op.insertArgument(op.getNumArguments(), pass->wordPtrType(),
                        DictionaryAttr::get(ctx));

      pass->setEffectDefs(op, op.getArgument(op.getNumArguments() - 2),
                          op.getArgument(op.getNumArguments() - 1));
    }

    _.setInsertionPointToStart(&*op.body().begin());

    auto buf_args =
        pass->effectAnalysis()->flattenedBufferArguments(op.getName());

    size_t arg_delta = 0;
    for (auto arg : buf_args) {
      auto arg_pos = arg + arg_delta++;
      auto old_arg = op.getArgument(arg_pos);

      op.insertArgument(arg_pos + 1, pass->bytePtrType(),
                        DictionaryAttr::get(ctx));
      op.setArgAttr(arg_pos + 1, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(ctx));

      op.insertArgument(arg_pos + 2, pass->wordType(),
                        DictionaryAttr::get(ctx));

      auto ptr = op.getArgument(arg_pos + 1);
      auto size = op.getArgument(arg_pos + 2);

      Value buf_struct = _.create<LLVM::UndefOp>(loc, pass->boundedBufType());
      buf_struct = _.create<LLVM::InsertValueOp>(  //
          loc, buf_struct, ptr, _.getI64ArrayAttr(0));
      buf_struct = _.create<LLVM::InsertValueOp>(  //
          loc, buf_struct, size, _.getI64ArrayAttr(1));

      old_arg.replaceAllUsesWith(buf_struct);
      op.eraseArgument(arg_pos);
    }

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
  auto src_type = op.src().getType().cast<ValueType>();
  auto dst_type = op.dst().getType().cast<ValueType>();

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

    _.replaceOp(op, operands[2]);
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

  // LLVM SwitchOp does not support 0 cases, so it must be handled explicitly.
  if (op.cases().empty()) {
    _.create<LLVM::BrOp>(loc, ValueRange{}, op.dflt());
    _.eraseOp(op);
    return success();
  }

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
                           /*defaultDestination=*/op.dflt(),
                           /*defaultOperands=*/ValueRange{},
                           /*caseValues=*/case_vals,
                           /*caseDestinations=*/op.cases(),
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

struct IndexOpLowering : public OpConversionPattern<IndexOp> {
  IndexOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                  LLVMGenPass* pass)
      : OpConversionPattern<IndexOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(IndexOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult IndexOpLowering::matchAndRewrite(
    IndexOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  auto src_type = op.seq().getType();

  if (auto ary = src_type.dyn_cast<ArrayType>()) {
    Value offset = _.create<MulOp>(
        loc, operands[1],
        pass->buildWordConstant(loc, _, ary->elem_size.bytes()));
    Value val = _.create<GEPOp>(loc, pass->bytePtrType(), operands[0], offset);
    _.replaceOp(op, val);
    return success();
  }

  return failure();
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

  pass->setEffectDefs(op, check_alloc, callback_alloc);

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
  auto loc = op.getLoc();

  const auto& buf_args =
      pass->effectAnalysis()->flattenedBufferArguments(op.callee());

  if (!pass->effectAnalysis()->hasEffects(op) && buf_args.empty()) {
    return failure();
  }

  llvm::SmallVector<Value, 4> new_operands;
  for (size_t i = 0, j = 0; i < operands.size(); ++i) {
    if (j < buf_args.size() && i == buf_args[j]) {
      auto [buf_ptr, buf_size] =
          pass->buildBoundedBufferDestructuring(loc, _, operands[i]);
      new_operands.push_back(buf_ptr);
      new_operands.push_back(buf_size);
      ++j;
    } else {
      new_operands.push_back(operands[i]);
    }
  }

  if (pass->effectAnalysis()->hasEffects(op)) {
    auto [check, cb] = pass->getEffectDefsFor(op);
    new_operands.insert(new_operands.end(), {check, cb});
  }

  auto call = _.create<mlir::CallOp>(op.getLoc(), op.getResultTypes(),
                                     op.callee(), new_operands);
  if (call.getNumResults() > 0) {
    _.replaceOp(op, call.getResults());
  } else {
    _.eraseOp(op);
  }

  return success();
}

struct DefaultOpLowering : public OpConversionPattern<DefaultOp> {
  DefaultOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                    LLVMGenPass* pass)
      : OpConversionPattern<DefaultOp>(converter, ctx) {}

  LogicalResult matchAndRewrite(DefaultOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult DefaultOpLowering::matchAndRewrite(
    DefaultOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  // All other types should have been lowered prior.
  auto type = op.dst().getType().cast<IntType>();

  Value ptr =
      _.create<BitcastOp>(loc, LLVMPointerType::get(type.toMLIR()), op.dst());
  auto zero = _.create<LLVM::ConstantOp>(loc, type.toMLIR(),
                                         _.getIntegerAttr(type.toMLIR(), 0));
  _.create<StoreOp>(loc, zero, ptr, type->alignment.bytes());

  _.eraseOp(op);
  return success();
}

struct UnitOpLowering : public OpConversionPattern<UnitOp> {
  UnitOpLowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,
                 LLVMGenPass* pass)
      : OpConversionPattern<UnitOp>(converter, ctx), pass(pass) {}

  LogicalResult matchAndRewrite(UnitOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass;
};

LogicalResult UnitOpLowering::matchAndRewrite(
    UnitOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto zero = pass->buildWordConstant(loc, _, 0);
  auto ptr = _.create<IntToPtrOp>(loc, pass->bytePtrType(), zero);
  _.replaceOp(op, ptr.getResult());
  return success();
}

void LLVMGenPass::runOnOperation() {
  auto* ctx = &getContext();

  effect_analysis_ = &getAnalysis<SideEffectAnalysis>();

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVMDialect>();
  target.addLegalOp<ModuleOp>();

  OwningRewritePatternList patterns(ctx);
  ProtoJitTypeConverter type_converter(ctx, this);
  populateStdToLLVMConversionPatterns(type_converter, patterns);
  patterns.add<CallOpLowering, FuncOpLowering, InvokeCallbackOpLowering,
               DecodeCatchOpLowering, ProjectOpLowering, TranscodeOpLowering,
               TagOpLowering, MatchOpLowering, SetCallbackOpLowering,
               DefaultOpLowering, UnitOpLowering, IndexOpLowering>(
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

}  // namespace

std::unique_ptr<Pass> createLLVMGenPass(const llvm::TargetMachine* machine) {
  return std::make_unique<LLVMGenPass>(machine);
}

}  // namespace pj
