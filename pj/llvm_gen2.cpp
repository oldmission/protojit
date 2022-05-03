#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include <llvm/Target/TargetMachine.h>

#include "defer.hpp"
#include "ir.hpp"
#include "llvm_extra.hpp"
#include "side_effect_analysis.hpp"
#include "util.hpp"

namespace pj {
using namespace mlir;
using mlir::CallOp;

using namespace mlir::LLVM;
using namespace ir;
using namespace types;

#define FOR_EACH_OP_LOWERING(V) \
  V(CallOp)                     \
  V(FuncOp)                     \
  V(InvokeCallbackOp)           \
  V(DecodeCatchOp)              \
  V(ProjectOp)                  \
  V(TranscodePrimitiveOp)       \
  V(TagOp)                      \
  V(MatchOp)                    \
  V(SetCallbackOp)              \
  V(DefaultOp)                  \
  V(UnitOp)                     \
  V(AllocateOp)                 \
  V(LengthOp)                   \
  V(StoreLengthOp)              \
  V(StoreRefOp)                 \
  V(ArrayIndexOp)               \
  V(VectorIndexOp)              \
  V(CopyTagOp)                  \
  V(PoisonOp)                   \
  V(SizeOp)

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

  mlir::Value buildTrueVal(mlir::Location& loc, mlir::OpBuilder& _) {
    return buildIntConstant(loc, _, Bits(1), 1);
  }

  mlir::Value buildFalseVal(mlir::Location& loc, mlir::OpBuilder& _) {
    return buildIntConstant(loc, _, Bits(1), 0);
  }

  auto ptrType(Width width) {
    return mlir::LLVM::LLVMPointerType::get(intType(width));
  }

  auto bytePtrType() { return ptrType(Bytes(1)); }

  auto wordPtrType() { return ptrType(wordSize()); }

  mlir::Type boundedBufType() {
    if (!bounded_buf_type_) {
      bounded_buf_type_ = mlir::LLVM::LLVMStructType::getNewIdentified(
          &getContext(), "!pj.bbuf", {bytePtrType(), wordType()});
    }
    return bounded_buf_type_;
  }

  mlir::Type rawBufType() { return bytePtrType(); }

  mlir::Type dummyBufType() { return wordType(); }

  Value buildOffsetPtr(Location loc, ConversionPatternRewriter& _, Value src,
                       Value offset, Width width) {
    auto ptr = _.create<GEPOp>(loc, bytePtrType(), src, offset);
    return _.create<BitcastOp>(loc, ptrType(width), ptr);
  }

  Value buildOffsetPtr(Location loc, ConversionPatternRewriter& _, Value src,
                       Width offset, Width width) {
    return buildOffsetPtr(loc, _, src,
                          buildWordConstant(loc, _, offset.bytes()), width);
  }

  Value buildTagPtr(Location loc, ConversionPatternRewriter& _,
                    VariantType type, Value variant) {
    return buildOffsetPtr(loc, _, variant, type.tag_offset(), type.tag_width());
  }

  Value buildBoundedBuf(Location loc, ConversionPatternRewriter& _, Value ptr,
                        Value size) {
    Value buf_struct = _.create<LLVM::UndefOp>(loc, boundedBufType());
    buf_struct = _.create<LLVM::InsertValueOp>(loc, buf_struct, ptr,
                                               _.getI64ArrayAttr(0));
    buf_struct = _.create<LLVM::InsertValueOp>(loc, buf_struct, size,
                                               _.getI64ArrayAttr(1));
    return buf_struct;
  }

  std::pair<Value, Value> buildBoundedBufferDestructuring(
      Location loc, ConversionPatternRewriter& _, Value buf);

  Value getBufPtr(Location loc, ConversionPatternRewriter& _, Value buf);

  LLVMGenPass(const llvm::TargetMachine* machine) : machine_(machine) {}

  ModuleOp module() { return ModuleOp{getOperation()}; }

  LLVM::GlobalOp buildGlobalConstant(Location loc,
                                     OpBuilder::Listener* listener,
                                     mlir::Type type, mlir::Attribute value) {
    auto name = "#const_" + std::to_string(const_suffix_++);
    auto _ = mlir::OpBuilder::atBlockBegin(module().getBody());
    _.setListener(listener);
    return _.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/false,
                                    LLVM::Linkage::Private, name, value);
  }

  SideEffectAnalysis* effectAnalysis() { return effect_analysis_; };

 private:
  size_t const_suffix_ = 0;
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
    addConversion([=](UserStateType type) { return pass->bytePtrType(); });
    addConversion(
        [=](BoundedBufferType type) { return pass->boundedBufType(); });
    addConversion([=](RawBufferType type) { return pass->rawBufType(); });
    addConversion([=](DummyBufferType type) { return pass->dummyBufType(); });
  }

  LLVMGenPass* pass;
};

#define DEFINE_OP_LOWERING(OP)                                               \
  struct OP##Lowering : public OpConversionPattern<OP> {                     \
    OP##Lowering(ProtoJitTypeConverter& converter, MLIRContext* ctx,         \
                 LLVMGenPass* pass)                                          \
        : OpConversionPattern<OP>(converter, ctx), pass(pass) {}             \
                                                                             \
    LogicalResult matchAndRewrite(OP op, ArrayRef<Value> operands,           \
                                  ConversionPatternRewriter& _) const final; \
                                                                             \
    LLVMGenPass* const pass;                                                 \
  };

FOR_EACH_OP_LOWERING(DEFINE_OP_LOWERING)

#undef DEFINE_OP_LOWERING

std::pair<Value, Value> LLVMGenPass::buildBoundedBufferDestructuring(
    Location loc, ConversionPatternRewriter& _, Value buf) {
  return {
      _.create<LLVM::ExtractValueOp>(  //
          loc, bytePtrType(), buf, _.getI64ArrayAttr(0)),
      _.create<LLVM::ExtractValueOp>(  //
          loc, wordType(), buf, _.getI64ArrayAttr(1)),
  };
}

Value LLVMGenPass::getBufPtr(Location loc, ConversionPatternRewriter& _,
                             Value buf) {
  if (buf.getType() == boundedBufType()) {
    return buildBoundedBufferDestructuring(loc, _, buf).first;
  } else if (buf.getType() == rawBufType()) {
    return buf;
  } else if (buf.getType() == dummyBufType()) {
    return _.create<LLVM::NullOp>(loc, bytePtrType());
  }
  UNREACHABLE();
}

LogicalResult FuncOpLowering::matchAndRewrite(
    FuncOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
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

    op.insertArgument(arg_pos + 2, pass->wordType(), DictionaryAttr::get(ctx));

    auto ptr = op.getArgument(arg_pos + 1);
    auto size = op.getArgument(arg_pos + 2);

    old_arg.replaceAllUsesWith(pass->buildBoundedBuf(loc, _, ptr, size));
    op.eraseArgument(arg_pos);
  }

  _.finalizeRootUpdate(op);
  return success();
}

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

LogicalResult TranscodePrimitiveOpLowering::matchAndRewrite(
    TranscodePrimitiveOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto src_type = op.src().getType().cast<ValueType>();
  auto dst_type = op.dst().getType().cast<ValueType>();

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

  return failure();
}

LogicalResult TagOpLowering::matchAndRewrite(
    TagOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  auto var_type = op.dst().getType().cast<VariantType>();

  auto store_ptr = pass->buildTagPtr(loc, _, var_type, operands[0]);
  auto tag_cst = pass->buildIntConstant(loc, _, var_type.tag_width(), op.tag());

  _.create<StoreOp>(op.getLoc(), tag_cst, store_ptr);

  _.eraseOp(op);
  return success();
}

LogicalResult CopyTagOpLowering::matchAndRewrite(
    CopyTagOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  Value src = operands[0], dst = operands[1];

  auto src_type = op.src().getType().cast<InlineVariantType>();
  auto dst_type = op.dst().getType().cast<InlineVariantType>();
  auto src_terms = src_type->terms;

  llvm::StringMap<const Term*> dst_terms;
  for (auto& term : dst_type->terms) {
    dst_terms[term.name] = &term;
  }

  Value src_tag_ptr = pass->buildTagPtr(loc, _, src_type, src);
  Value dst_tag_ptr = pass->buildTagPtr(loc, _, dst_type, dst);

  Value src_tag = _.create<LoadOp>(loc, src_tag_ptr);

  bool exact_match = true;

  std::vector<std::pair<uint64_t, uint64_t>> tag_mapping{{0, 0}};
  for (auto& term : src_terms) {
    if (auto it = dst_terms.find(term.name); it != dst_terms.end()) {
      exact_match &= it->second->tag == term.tag;
      tag_mapping.emplace_back(term.tag, it->second->tag);
    } else {
      exact_match = false;
      tag_mapping.emplace_back(term.tag, 0);
    }
  }

  if (exact_match) {
    if (src_type->tag_width < dst_type->tag_width) {
      src_tag =
          _.create<ZExtOp>(loc, pass->intType(dst_type->tag_width), src_tag);
    } else if (src_type->tag_width > dst_type->tag_width) {
      src_tag =
          _.create<TruncOp>(loc, pass->intType(dst_type->tag_width), src_tag);
    }
    _.create<StoreOp>(loc, src_tag, dst_tag_ptr);
    _.eraseOp(op);
    return success();
  }

  // Build a table to look up the tag.
  std::sort(tag_mapping.begin(), tag_mapping.end(),
            [](auto& l, auto& r) { return l.first < r.first; });

  // TODO: this is int64_t whereas tags can be uint64_t
  llvm::SmallVector<int64_t> table;
  for (size_t i = 0, t = 0; i < tag_mapping.size(); ++t) {
    if (tag_mapping[i].first == t) {
      table.push_back(tag_mapping[i++].second);
    } else {
      table.push_back(0);
    }
  }

  auto too_high = _.create<ICmpOp>(
      loc, pass->intType(Bits(1)), ICmpPredicate::uge, src_tag,
      pass->buildIntConstant(loc, _, src_type->tag_width, table.size()));

  auto dst_tag = _.create<scf::IfOp>(loc, pass->intType(dst_type->tag_width),
                                     too_high, /*withElseRegion=*/true);
  _.create<StoreOp>(loc, dst_tag.getResult(0), dst_tag_ptr);

  // src_tag is in bounds, use the table.
  {
    _.setInsertionPointToStart(dst_tag.elseBlock());
    auto table_type =
        LLVMArrayType::get(pass->intType(dst_type->tag_width), table.size());
    auto table_cst = pass->buildGlobalConstant(loc, _.getListener(), table_type,
                                               _.getI64TensorAttr(table));
    Value table_ptr = _.create<LLVM::AddressOfOp>(
        loc, LLVMPointerType::get(table_type), table_cst.getName());
    table_ptr = _.create<BitcastOp>(loc, dst_tag_ptr.getType(), table_ptr);

    auto result_tag_ptr =
        _.create<GEPOp>(loc, table_ptr.getType(), table_ptr, src_tag);
    Value result = _.create<LoadOp>(loc, result_tag_ptr);
    _.create<scf::YieldOp>(loc, result);
  }

  // src_tag is not in bounds, use undef.
  {
    _.setInsertionPointToStart(dst_tag.thenBlock());
    Value undef_tag = pass->buildIntConstant(loc, _, dst_type->tag_width, 0);
    _.create<scf::YieldOp>(loc, undef_tag);
  }

  _.eraseOp(op);
  return success();
}

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

LogicalResult ArrayIndexOpLowering::matchAndRewrite(
    ArrayIndexOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  Value offset = _.create<MulOp>(
      loc, operands[1],
      pass->buildWordConstant(
          loc, _, op.arr().getType().cast<ArrayType>()->elem_size.bytes()));
  Value val = _.create<GEPOp>(loc, pass->bytePtrType(), operands[0], offset);
  _.replaceOp(op, val);
  return success();
}

LogicalResult VectorIndexOpLowering::matchAndRewrite(
    VectorIndexOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto type = op.vec().getType().cast<VectorType>();

  Value start;
  switch (op.region()) {
    case VectorRegion::Inline:
      start = _.create<GEPOp>(
          loc, pass->bytePtrType(), operands[0],
          pass->buildWordConstant(loc, _, type->inline_payload_offset.bytes()));
      break;
    case VectorRegion::Partial:
      start =
          _.create<GEPOp>(loc, pass->bytePtrType(), operands[0],
                          pass->buildWordConstant(
                              loc, _, type->partial_payload_offset.bytes()));
      break;
    case VectorRegion::Reference: {
      auto ref_ptr = pass->buildOffsetPtr(loc, _, operands[0], type->ref_offset,
                                          type->ref_size);
      auto ref = _.create<LoadOp>(loc, ref_ptr);

      if (type->reference_mode == Vector::kOffset) {
        // TODO: bounds check on the offset
        start = _.create<GEPOp>(loc, pass->bytePtrType(), operands[0],
                                ValueRange{ref});
      } else {
        assert(type->ref_size == pass->wordSize());
        start = _.create<IntToPtrOp>(loc, pass->bytePtrType(), ref);
      }
    } break;
    case VectorRegion::Buffer:
      start = pass->getBufPtr(loc, _, operands[2]);
      break;
  }

  Value offset = _.create<MulOp>(
      loc, operands[1],
      pass->buildWordConstant(
          loc, _, op.vec().getType().cast<VectorType>()->elemSize().bytes()));
  Value val = _.create<GEPOp>(loc, pass->bytePtrType(), start, offset);

  _.replaceOp(op, val);
  return success();
}

LogicalResult AllocateOpLowering::matchAndRewrite(
    AllocateOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto buf = operands[0];
  if (buf.getType() == pass->boundedBufType()) {
    auto [ptr, size] = pass->buildBoundedBufferDestructuring(loc, _, buf);
    // TODO: insert a bounds check
    ptr = _.create<GEPOp>(loc, pass->bytePtrType(), ptr, operands[1]);
    size = _.create<SubOp>(loc, size, operands[1]);
    buf = pass->buildBoundedBuf(loc, _, ptr, size);
  } else if (buf.getType() == pass->rawBufType()) {
    buf = _.create<GEPOp>(loc, pass->bytePtrType(), buf, operands[1]);
  } else if (buf.getType() == pass->dummyBufType()) {
    buf = _.create<AddOp>(loc, pass->wordType(), buf, operands[1]);
  } else {
    UNREACHABLE();
  }
  _.replaceOp(op, buf);
  return success();
}

LogicalResult LengthOpLowering::matchAndRewrite(
    LengthOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto type = op.vec().getType().cast<VectorType>();

  if (type->max_length == 0) {
    _.replaceOp(op, pass->buildWordConstant(loc, _, 0));
    return success();
  }

  Value length_ptr = pass->buildOffsetPtr(
      loc, _, operands[0], type->length_offset, type->length_size);
  Value length = _.create<ZExtOp>(loc, pass->wordType(),
                                  _.create<LoadOp>(loc, length_ptr));
  _.replaceOp(op, length);
  return success();
}

LogicalResult StoreLengthOpLowering::matchAndRewrite(
    StoreLengthOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto type = op.vec().getType().cast<VectorType>();

  Value length_ptr = pass->buildOffsetPtr(
      loc, _, operands[0], type->length_offset, type->length_size);
  _.create<StoreOp>(
      loc,
      _.create<TruncOp>(loc, pass->intType(type->length_size), operands[1]),
      length_ptr);
  _.eraseOp(op);
  return success();
}

LogicalResult StoreRefOpLowering::matchAndRewrite(
    StoreRefOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto type = op.vec().getType().cast<VectorType>();
  auto buf_as_int = _.create<PtrToIntOp>(loc, pass->wordType(),
                                         pass->getBufPtr(loc, _, operands[1]));
  auto ref_ptr = pass->buildOffsetPtr(loc, _, operands[0], type->ref_offset,
                                      type->ref_size);

  if (type->reference_mode == Vector::kPointer) {
    assert(type->ref_size == pass->wordSize());
    _.create<StoreOp>(loc, buf_as_int, ref_ptr);
  } else {
    auto vec_as_int = _.create<PtrToIntOp>(loc, pass->wordType(), operands[0]);
    auto offset =
        _.create<SubOp>(loc, pass->wordType(), buf_as_int, vec_as_int);
    // TODO: debug assert that this fits within ref_size
    auto offset_trunc =
        _.create<TruncOp>(loc, pass->intType(type->ref_size), offset);
    _.create<StoreOp>(loc, offset_trunc, ref_ptr);
  }

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

  pass->setEffectDefs(op, check_alloc, callback_alloc);

  // Initialize the allocations.
  auto zero = pass->buildWordConstant(loc, _, 0);
  _.create<StoreOp>(loc, zero, check_alloc);
  _.create<StoreOp>(loc, zero, callback_alloc);

  // Replace uses of the op with the YieldOp value.
  auto* yield_op = &op.body().front().back();
  assert(isa<YieldOp>(yield_op));

  auto yield_val = cast<YieldOp>(yield_op).result();

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

LogicalResult DefaultOpLowering::matchAndRewrite(
    DefaultOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  if (auto type = op.dst().getType().dyn_cast<IntType>()) {
    Value ptr =
        _.create<BitcastOp>(loc, LLVMPointerType::get(type.toMLIR()), op.dst());
    auto zero = _.create<LLVM::ConstantOp>(loc, type.toMLIR(),
                                           _.getIntegerAttr(type.toMLIR(), 0));
    _.create<StoreOp>(loc, zero, ptr, type->alignment.bytes());

    _.eraseOp(op);
    return success();
  }

  if (auto type = op.dst().getType().dyn_cast<VectorType>()) {
    auto zero = pass->buildWordConstant(loc, _, 0);
    _.create<StoreLengthOp>(loc, op.dst(), zero);

    _.eraseOp(op);
    return success();
  }

  // All other types should have been lowered prior.
  return failure();
}

LogicalResult UnitOpLowering::matchAndRewrite(
    UnitOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto zero = pass->buildWordConstant(loc, _, 0);
  auto ptr = _.create<IntToPtrOp>(loc, pass->bytePtrType(), zero);
  _.replaceOp(op, ptr.getResult());
  return success();
}

LogicalResult PoisonOpLowering::matchAndRewrite(
    PoisonOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto width = pass->buildWordConstant(loc, _, op.width().bytes());
  auto ptr = pass->buildOffsetPtr(loc, _, operands[0], op.offset(), Bytes(1));
  _.create<LLVM::LifetimeEndOp>(loc, width, ptr);
  _.eraseOp(op);
  return success();
}

LogicalResult SizeOpLowering::matchAndRewrite(
    SizeOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  // Get the result buffer from the SizeOp body
  auto* yield_op = &op.body().front().back();
  assert(isa<YieldOp>(yield_op));

  auto yield_val = cast<YieldOp>(yield_op).result();
  _.eraseOp(yield_op);

  // Inline the body
  auto* start = _.getInsertionBlock();
  auto* body_entry = &op.body().front();
  auto* continuation = _.splitBlock(start, _.getInsertionPoint());
  _.inlineRegionBefore(op.body(), continuation);

  {
    _.setInsertionPointToEnd(start);
    auto zero = pass->buildWordConstant(loc, _, 0);
    _.create<mlir::BranchOp>(loc, body_entry, ValueRange{zero});
  }

  _.setInsertionPointToEnd(body_entry);
  _.create<mlir::BranchOp>(loc, continuation, ValueRange{});

  {
    _.setInsertionPointToStart(continuation);
    _.replaceOp(op, yield_val);
  }

  return success();
}

void LLVMGenPass::runOnOperation() {
  auto* ctx = &getContext();

  effect_analysis_ = &getAnalysis<SideEffectAnalysis>();

  ConversionTarget target(*ctx);

  target.addLegalDialect<LLVMDialect>();
  target.addLegalDialect<LLVMExtraDialect>();

  target.addLegalOp<ModuleOp>();

  OwningRewritePatternList patterns(ctx);
  ProtoJitTypeConverter type_converter(ctx, this);
  populateStdToLLVMConversionPatterns(type_converter, patterns);
  populateLoopToStdConversionPatterns(patterns);

#define REGISTER_OP_LOWERING(V) \
  patterns.add<V##Lowering>(type_converter, ctx, this);
  FOR_EACH_OP_LOWERING(REGISTER_OP_LOWERING)
#undef REGISTER_OP_LOWERING

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
