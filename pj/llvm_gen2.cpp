#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
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
  V(ThrowOp)                    \
  V(SetCallbackOp)              \
  V(DefaultOp)                  \
  V(UnitOp)                     \
  V(AlignOp)                    \
  V(AllocateOp)                 \
  V(LengthOp)                   \
  V(StoreLengthOp)              \
  V(StoreRefOp)                 \
  V(ArrayIndexOp)               \
  V(VectorIndexOp)              \
  V(CopyTagOp)                  \
  V(PoisonOp)                   \
  V(SizeOp)                     \
  V(ReflectOp)                  \
  V(AssumeOp)                   \
  V(DefineProtocolOp)

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

  mlir::IntegerType wordType() { return intType(wordSize()); }

  mlir::IntegerType boolType() { return intType(Bits(1)); }

  mlir::IntegerType intType(Width width) {
    return mlir::IntegerType::get(&getContext(), width.bits(),
                                  mlir::IntegerType::Signless);
  }

  mlir::Attribute intAttr(Width width, size_t value) {
    return mlir::IntegerAttr::get(intType(width), value);
  }

  mlir::Attribute wordAttr(size_t value) {
    return mlir::IntegerAttr::get(intType(wordSize()), value);
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

  Value buildBoundedBuf(Location loc, ConversionPatternRewriter& _,
                        std::optional<Value> ptr, std::optional<Value> size) {
    Value buf_struct = _.create<LLVM::UndefOp>(loc, boundedBufType());
    if (ptr.has_value()) {
      buf_struct = _.create<LLVM::InsertValueOp>(loc, buf_struct, ptr.value(),
                                                 _.getI64ArrayAttr(0));
    }
    if (size.has_value()) {
      buf_struct = _.create<LLVM::InsertValueOp>(loc, buf_struct, size.value(),
                                                 _.getI64ArrayAttr(1));
    }
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
    addConversion([=](HandlersArrayType type) { return pass->wordPtrType(); });
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
    LogicalResult matchAndRewrite(OP op, llvm::ArrayRef<Value> operands,     \
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
  }
  if (buf.getType() == rawBufType()) {
    return buf;
  }
  if (buf.getType() == dummyBufType()) {
    return _.create<LLVM::NullOp>(loc, bytePtrType());
  }
  UNREACHABLE();
}

LogicalResult FuncOpLowering::matchAndRewrite(
    FuncOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto* ctx = _.getContext();
  auto loc = op.getLoc();

  // 1. Create a FuncOp with the new signature.

  llvm::SmallVector<mlir::Type, 4> new_fn_types;
  llvm::SmallVector<DictionaryAttr, 4> new_fn_attrs;

  auto buf_args =
      pass->effectAnalysis()->flattenedBufferArguments(op.getName());

  NamedAttribute no_alias_attr{
      mlir::Identifier::get(LLVM::LLVMDialect::getNoAliasAttrName(), ctx),
      UnitAttr::get(ctx)};

  for (intptr_t i = 0, j = 0; i < op.getNumArguments(); ++i) {
    if (j < buf_args.size() && i == buf_args[j]) {
      // pointer
      new_fn_types.push_back(pass->bytePtrType());
      new_fn_attrs.push_back(DictionaryAttr::get(ctx, {no_alias_attr}));

      // size
      new_fn_types.push_back(pass->wordType());
      new_fn_attrs.push_back(DictionaryAttr::get(ctx, {}));
      ++j;
    } else {
      new_fn_types.push_back(op.getArgument(i).getType());
      new_fn_attrs.push_back(DictionaryAttr::get(ctx, op.getArgAttrs(i)));
    }
  }

  if (pass->effectAnalysis()->hasEffects(op)) {
    // success
    new_fn_types.push_back(pass->ptrType(Bits(1)));
    new_fn_attrs.push_back(DictionaryAttr::get(ctx, {}));

    // callback
    new_fn_types.push_back(pass->wordPtrType());
    new_fn_attrs.push_back(DictionaryAttr::get(ctx, {}));
  }

  auto func = mlir::FuncOp::create(
      loc, op.getName(),
      _.getFunctionType(new_fn_types, op.getType().getResults()), {},
      new_fn_attrs);
  _.notifyOperationInserted(func);

  auto *new_entry = func.addEntryBlock(), *old_entry = &op.getBlocks().front();

  // 2. Update argument uses to refrence args from the new function.

  if (pass->effectAnalysis()->hasEffects(op)) {
    pass->setEffectDefs(op, func.getArgument(func.getNumArguments() - 2),
                        func.getArgument(func.getNumArguments() - 1));
  }

  {
    auto& fn_blocks = func.body().getBlocks();
    auto& op_blocks = op.body().getBlocks();
    fn_blocks.splice(fn_blocks.end(), op_blocks, ++op_blocks.begin(),
                     op_blocks.end());
  }

  while (!old_entry->empty()) {
    old_entry->front().moveBefore(new_entry, new_entry->end());
  }

  _.setInsertionPointToStart(new_entry);

  for (intptr_t i = 0, j = 0, k = 0; i < op.getNumArguments(); ++i) {
    if (j < buf_args.size() && i == buf_args[j]) {
      auto old_arg = op.getArgument(i);
      auto ptr = func.getArgument(k), size = func.getArgument(k + 1);
      auto to = pass->buildBoundedBuf(loc, _, ptr, size);
      _.replaceUsesOfBlockArgument(old_arg, to);
      k += 2, ++j;
    } else {
      _.replaceUsesOfBlockArgument(op.getArgument(i), func.getArgument(k++));
    }
  }

  _.eraseOp(op);
  pass->module().push_back(func);
  _.notifyOperationInserted(func);
  return success();
}

LogicalResult ProjectOpLowering::matchAndRewrite(
    ProjectOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto base = operands[0];
  auto base_type = op.base().getType();
  auto result = op.getResult().getType();

  if (base_type.isa<types::ValueType>() ||
      base_type.isa<types::RawBufferType>()) {
    // ValueTypes and RawBuffers are represented as 'char*' in LLVM.
    // The output should have the same representation. We can't create
    // a bounded buffer anyway without some reference for the size.
    ASSERT(result.isa<types::ValueType>() ||
           result.isa<types::RawBufferType>());

    Value val =
        _.create<GEPOp>(loc, pass->bytePtrType(), base,
                        pass->buildWordConstant(loc, _, op.offset().bytes()));

    if (result.isa<types::ValueType>() && op.frozen()) {
      _.create<LLVM::InvariantStartOp>(
          loc,
          pass->buildWordConstant(
              loc, _, result.cast<types::ValueType>().headSize().bytes()),
          val);
    }

    _.replaceOp(op, val);
  } else if (base_type.isa<types::BoundedBufferType>() &&
             (result.isa<types::ValueType>() ||
              result.isa<types::RawBufferType>())) {
    auto [buf, __] = pass->buildBoundedBufferDestructuring(loc, _, base);
    Value val =
        _.create<GEPOp>(loc, pass->bytePtrType(), buf,
                        pass->buildWordConstant(loc, _, op.offset().bytes()));
    _.replaceOp(op, val);
  } else {
    UNREACHABLE();
  }

  return success();
}

LogicalResult TranscodePrimitiveOpLowering::matchAndRewrite(
    TranscodePrimitiveOp op, llvm::ArrayRef<Value> operands,
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
      if (src->sign == Sign::kSigned) {
        val = _.create<SExtOp>(loc, dst.toMLIR(), val);
      } else {
        val = _.create<ZExtOp>(loc, dst.toMLIR(), val);
      }
    } else {
      val = _.create<TruncOp>(loc, dst.toMLIR(), val);
    }

    ASSERT(llvm::isPowerOf2_64(dst->alignment.bytes()));
    _.create<StoreOp>(op.getLoc(), val, dst_ptr, dst->alignment.bytes());

    _.eraseOp(op);
    return success();
  }

  return failure();
}

LogicalResult TagOpLowering::matchAndRewrite(
    TagOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  auto var_type = op.dst().getType().cast<VariantType>();

  auto store_ptr = pass->buildTagPtr(loc, _, var_type, operands[0]);
  auto tag_cst = pass->buildIntConstant(loc, _, var_type.tag_width(), op.tag());

  _.create<StoreOp>(op.getLoc(), tag_cst, store_ptr);

  _.eraseOp(op);
  return success();
}

LogicalResult CopyTagOpLowering::matchAndRewrite(
    CopyTagOp op, llvm::ArrayRef<Value> operands,
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

  // Tags are uint64_t, but this is fine because LLVM treats signed and unsigned
  // ints the same, except for signed instructions, which are not used here.
  llvm::SmallVector<int64_t> table;
  for (size_t i = 0, t = 0; i < tag_mapping.size(); ++t) {
    if (tag_mapping[i].first == t) {
      table.push_back(tag_mapping[i++].second);
    } else {
      table.push_back(0);
    }
  }

  auto too_high = _.create<ICmpOp>(
      loc, pass->boolType(), ICmpPredicate::uge, src_tag,
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
    MatchOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
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

  tag_val = _.create<ZExtOp>(op.getLoc(), pass->intType(Bits(64)), tag_val);

  llvm::SmallVector<llvm::APInt, 4> case_vals;
  for (auto& term : var_type.terms()) {
    case_vals.emplace_back(64, term.tag);
  }
  std::sort(case_vals.begin(), case_vals.end(),
            [](const APInt& a, const APInt& b) { return a.ult(b); });

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
    InvokeCallbackOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  llvm::SmallVector<Type, 2> dispatch_args = {
      operands[0].getType(),
      operands[2].getType(),
  };
  auto void_ty = LLVM::LLVMVoidType::get(_.getContext());
  auto dispatch_fn_type = LLVM::LLVMFunctionType::get(void_ty, dispatch_args);
  auto dispatch_type = LLVM::LLVMPointerType::get(dispatch_fn_type, 0);

  auto [__, callback_store] = pass->getEffectDefsFor(op);

  Value callback = _.create<LLVM::LoadOp>(loc, callback_store);

  // Check if the callback is set.
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

  // Handler indices stored in the slot are +1, so 0 represents no handler.
  callback =
      _.create<LLVM::SubOp>(loc, callback, pass->buildWordConstant(loc, _, 1));
  callback =
      _.create<LLVM::GEPOp>(loc, pass->wordPtrType(), operands[1], callback);
  callback = _.create<LLVM::LoadOp>(loc, callback);
  callback = _.create<LLVM::IntToPtrOp>(loc, dispatch_type, callback);
  auto call = _.create<LLVM::CallOp>(
      loc, TypeRange{LLVM::LLVMVoidType::get(_.getContext())},
      ValueRange{callback, operands[0], operands[2]});
  ASSERT(call.verify().succeeded());
  _.create<LLVM::BrOp>(loc, ValueRange{}, end_block);

  _.eraseOp(op);
  _.setInsertionPointToStart(end_block);
  return success();
}

LogicalResult ThrowOpLowering::matchAndRewrite(
    ThrowOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  auto [check_store, __] = pass->getEffectDefsFor(op);
  _.create<LLVM::StoreOp>(loc, pass->buildTrueVal(loc, _), check_store);
  _.create<LLVM::ReturnOp>(loc,
                           ValueRange{pass->buildBoundedBuf(loc, _, {}, {})});

  _.eraseOp(op);
  return success();
}

LogicalResult SetCallbackOpLowering::matchAndRewrite(
    SetCallbackOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto [__, callback_store] = pass->getEffectDefsFor(op);

  // We set the handler index +1 so that 0 denotes no handler being set.
  auto target = pass->buildWordConstant(loc, _, op.target().getZExtValue() + 1);
  _.create<LLVM::StoreOp>(loc, target, callback_store);

  _.eraseOp(op);
  return success();
}

LogicalResult ArrayIndexOpLowering::matchAndRewrite(
    ArrayIndexOp op, llvm::ArrayRef<Value> operands,
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
    VectorIndexOp op, llvm::ArrayRef<Value> operands,
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

      if (type->reference_mode == ReferenceMode::kOffset) {
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

LogicalResult AlignOpLowering::matchAndRewrite(
    AlignOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  assert(op.alignment() == 1 || op.alignment() == 2 || op.alignment() == 4 ||
         op.alignment() == 8);

  auto buf = operands[0];

  if (op.alignment() == 1) {
    _.replaceOp(op, buf);
    return success();
  }

  auto loc = op.getLoc();
  auto buf_as_int =
      _.create<PtrToIntOp>(loc, pass->wordType(), pass->getBufPtr(loc, _, buf));

  // Compute ((ptr - 1) | (alignment - 1)) + 1 to align the pointer up
  auto ptr_minus_one =
      _.create<SubOp>(loc, buf_as_int, pass->buildWordConstant(loc, _, 1));
  auto or_alignment_minus_one = _.create<LLVM::OrOp>(
      loc, ptr_minus_one, pass->buildWordConstant(loc, _, op.alignment() - 1));
  auto aligned = _.create<AddOp>(loc, or_alignment_minus_one,
                                 pass->buildWordConstant(loc, _, 1));
  auto diff = _.create<SubOp>(loc, aligned, buf_as_int);

  auto alloc = _.create<AllocateOp>(loc, buf.getType(), buf, diff);

  pass->effectAnalysis()->replaceOperation(op, alloc);
  _.replaceOp(op, ValueRange{alloc});

  return success();
}

LogicalResult AllocateOpLowering::matchAndRewrite(
    AllocateOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto buf = operands[0];
  if (buf.getType() == pass->boundedBufType()) {
    auto* cur_block = _.getInsertionBlock();
    auto* continuation = _.splitBlock(cur_block, _.getInsertionPoint());
    auto* throw_block = _.createBlock(cur_block->getParent());

    _.setInsertionPointToEnd(cur_block);
    auto [ptr, size] = pass->buildBoundedBufferDestructuring(loc, _, buf);
    size = _.create<SubOp>(loc, size, operands[1]);

    auto too_small = _.create<ICmpOp>(loc, pass->boolType(), ICmpPredicate::slt,
                                      size, pass->buildWordConstant(loc, _, 0));
    _.create<LLVM::CondBrOp>(loc, too_small, throw_block, ValueRange{},
                             continuation, ValueRange{},
                             std::make_pair(0u, 1u));

    _.setInsertionPointToStart(throw_block);
    auto throw_op = _.create<ThrowOp>(loc);
    pass->effectAnalysis()->replaceOperation(op, throw_op);

    _.setInsertionPointToStart(continuation);
    ptr = _.create<GEPOp>(loc, pass->bytePtrType(), ptr, operands[1]);
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
    LengthOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
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

  if (type->max_length >= 0) {
    auto max = pass->buildWordConstant(loc, _, type->max_length);
    auto cond = _.create<ICmpOp>(loc, pass->intType(Bits(1)),
                                 ICmpPredicate::ule, length, max);
    _.create<LLVM::AssumeOp>(loc, cond);
  }

  _.replaceOp(op, length);
  return success();
}

LogicalResult StoreLengthOpLowering::matchAndRewrite(
    StoreLengthOp op, llvm::ArrayRef<Value> operands,
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
    StoreRefOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto type = op.vec().getType().cast<VectorType>();
  auto buf_as_int = _.create<PtrToIntOp>(loc, pass->wordType(),
                                         pass->getBufPtr(loc, _, operands[1]));
  auto ref_ptr = pass->buildOffsetPtr(loc, _, operands[0], type->ref_offset,
                                      type->ref_size);

  if (type->reference_mode == ReferenceMode::kPointer) {
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
    DecodeCatchOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto* ctx = _.getContext();
  auto loc = op.getLoc();

  // Create real definitions for the anchors and replace anchor uses with
  // them.
  auto one = pass->buildWordConstant(loc, _, 1);
  auto check_alloc = _.create<AllocaOp>(loc, pass->ptrType(Bits(1)), one);
  auto callback_alloc = _.create<AllocaOp>(loc, pass->wordPtrType(), one);

  pass->setEffectDefs(op, check_alloc, callback_alloc);

  // Initialize the allocations.
  _.create<StoreOp>(loc, pass->buildFalseVal(loc, _), check_alloc);
  _.create<StoreOp>(loc, pass->buildWordConstant(loc, _, 0), callback_alloc);

  // Restructure the blocks from
  //  ^start:
  //    <before DecodeCatchOp>
  //    %buf = "pj.catch"() ({ <body> })
  //    <after DecodeCatchOp>
  // to
  //  ^start:
  //    <before DecodeCatchOp>
  //    br ^body
  //  ^body:
  //    <contents of body>
  //    ERASED: "pj.yield"(%result_buf)
  //    br ^continuation
  //  ^continuation_start:
  //    %check = <load value of check>
  //    llvm.cond_br %check (weights 0, 1), ^fail, ^succ
  //  ^fail:
  //    br ^continuation_end(<null buffer>)
  //  ^succ:
  //    br ^continuation_end(%result_buf)
  //  ^continuation_end(%buf):
  //    <after DecodeCatchOp using block argument in place of original %buf>
  auto* start = _.getInsertionBlock();

  auto* body = &op.body().front();
  assert(op.body().hasOneBlock());

  // Get the result buffer contained in the YieldOp.
  auto* yield_op = &body->back();
  assert(isa<YieldOp>(yield_op));
  auto result_buf = cast<YieldOp>(yield_op).result();
  _.eraseOp(yield_op);

  auto* continuation_start = _.splitBlock(start, _.getInsertionPoint());
  auto* fail = _.createBlock(start->getParent());
  auto* succ = _.createBlock(start->getParent());

  _.inlineRegionBefore(op.body(), continuation_start);

  _.setInsertionPointToEnd(start);
  _.create<mlir::BranchOp>(loc, body, ValueRange{});

  _.setInsertionPointToEnd(body);
  _.create<mlir::BranchOp>(loc, continuation_start, ValueRange{});

  _.setInsertionPointToStart(continuation_start);
  auto check_val = _.create<LoadOp>(loc, pass->boolType(), check_alloc);
  _.create<LLVM::CondBrOp>(loc, check_val, fail, ValueRange{}, succ,
                           ValueRange{}, std::make_pair(0u, 1u));

  auto* continuation_end =
      _.splitBlock(continuation_start, _.getInsertionPoint());
  auto result_buf_arg =
      continuation_end->addArgument(BoundedBufferType::get(ctx), loc);

  _.setInsertionPointToStart(fail);
  auto empty_buf = pass->buildBoundedBuf(
      loc, _, _.create<LLVM::NullOp>(loc, pass->bytePtrType()), {});
  _.create<LLVM::BrOp>(loc, ValueRange{empty_buf}, continuation_end);

  _.setInsertionPointToStart(succ);
  _.create<LLVM::BrOp>(loc, ValueRange{result_buf}, continuation_end);

  _.replaceOp(op, result_buf_arg);

  return success();
}

LogicalResult CallOpLowering::matchAndRewrite(
    mlir::CallOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  const auto& buf_args =
      pass->effectAnalysis()->flattenedBufferArguments(op.callee());

  const auto* effects = pass->effectAnalysis();

  if (!effects->hasEffects(op) && buf_args.empty()) {
    return failure();
  }

  llvm::SmallVector<Value, 4> new_operands;
  for (size_t i = 0, j = 0; i < operands.size(); ++i) {
    if (j < buf_args.size() && i == buf_args[j]) {
      auto [buf_ptr, buf_size] =
          pass->buildBoundedBufferDestructuring(loc, _, operands[i]);
      new_operands.push_back(buf_ptr);
      new_operands.push_back(buf_size);
      j++;
    } else {
      new_operands.push_back(operands[i]);
    }
  }

  if (effects->hasEffects(op)) {
    auto [check, cb] = pass->getEffectDefsFor(op);
    new_operands.insert(new_operands.end(), {check, cb});
  }

  auto call = _.create<mlir::CallOp>(op.getLoc(), op.getResultTypes(),
                                     op.callee(), new_operands);

  if (effects->hasEffects(effects->effectProviderFor(op))) {
    auto* cur_block = _.getInsertionBlock();
    auto* continuation = _.splitBlock(cur_block, _.getInsertionPoint());
    auto* exit_block = _.createBlock(cur_block->getParent());

    _.setInsertionPointToEnd(cur_block);
    auto [check, __] = pass->getEffectDefsFor(op);
    auto check_val = _.create<LoadOp>(loc, check);
    _.create<LLVM::CondBrOp>(loc, check_val, exit_block, ValueRange{},
                             continuation, ValueRange{},
                             std::make_pair(0u, 1u));

    _.setInsertionPointToStart(exit_block);
    _.create<LLVM::ReturnOp>(loc,
                             ValueRange{pass->buildBoundedBuf(loc, _, {}, {})});
  }

  if (call.getNumResults() > 0) {
    _.replaceOp(op, call.getResults());
  } else {
    _.eraseOp(op);
  }

  return success();
}

LogicalResult DefaultOpLowering::matchAndRewrite(
    DefaultOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  if (auto type = op.dst().getType().dyn_cast<IntType>()) {
    Value ptr =
        _.create<BitcastOp>(loc, LLVMPointerType::get(type.toMLIR()), op.dst());
    auto zero = _.create<LLVM::ConstantOp>(loc, type.toMLIR(),
                                           _.getIntegerAttr(type.toMLIR(), 0));
    ASSERT(llvm::isPowerOf2_64(type->alignment.bytes()));
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
    UnitOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto null = _.create<LLVM::NullOp>(op.getLoc(), pass->bytePtrType());
  _.replaceOp(op, ValueRange{null});
  return success();
}

LogicalResult PoisonOpLowering::matchAndRewrite(
    PoisonOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();

  auto ptr = pass->buildOffsetPtr(loc, _, operands[0], op.offset(), Bytes(1));
  _.create<LLVM::MemsetOp>(
      loc, ptr, _.create<LLVM::UndefOp>(loc, pass->intType(Bytes(1))),
      pass->buildWordConstant(loc, _, op.width().bytes()),
      pass->buildFalseVal(loc, _));
  _.eraseOp(op);
  return success();
}

LogicalResult SizeOpLowering::matchAndRewrite(
    SizeOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
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

LogicalResult ReflectOpLowering::matchAndRewrite(
    ReflectOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  // TODO:
  // 1. Create binary representation of the source schema in the host's self
  // representation.
  // 2. Save the schema in the constant pool and get a pointer to it.
  // 3. Save pointers to the schema and object in the destination.

  _.eraseOp(op);
  return success();
}

LogicalResult AssumeOpLowering::matchAndRewrite(
    ir::AssumeOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  _.create<LLVM::AssumeOp>(op.getLoc(), operands[0]);
  _.eraseOp(op);
  return success();
}

LogicalResult DefineProtocolOpLowering::matchAndRewrite(
    ir::DefineProtocolOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto proto_cst_type =
      LLVMArrayType::get(pass->intType(Bytes(1)), op.proto().size());
  auto proto_cst_attr = StringAttr::get(_.getContext(), op.proto());
  _.setInsertionPointToStart(pass->module().getBody());
  _.create<LLVM::GlobalOp>(op.getLoc(), proto_cst_type, /*isConstant=*/false,
                           LLVM::Linkage::External, op.ptrName(),
                           proto_cst_attr);
  _.create<LLVM::GlobalOp>(op.getLoc(), pass->wordType(), /*isConstant=*/false,
                           LLVM::Linkage::External, op.sizeName(),
                           pass->wordAttr(op.proto().size()));
  _.eraseOp(op);
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
    return;
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
