#include <unordered_map>
#include <unordered_set>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "defer.hpp"
#include "ir.hpp"
#include "util.hpp"

namespace pj {
struct FnKey {
  types::ValueType from;
  types::ValueType to;
  mlir::Attribute attr;

  bool operator==(const FnKey& other) const {
    return from == other.from && to == other.to && attr == other.attr;
  }
};
}  // namespace pj

namespace std {
template <>
struct hash<pj::FnKey> {
  size_t operator()(const pj::FnKey& key) const {
    using ::llvm::hash_value;
    return llvm::hash_combine(hash_value(key.from), hash_value(key.to),
                              hash_value(key.attr));
  }
};
}  // namespace std

namespace pj {
using namespace mlir;
using namespace ir;
using namespace ir2;
using namespace types;

namespace {
struct GeneratePass
    : public PassWrapper<GeneratePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ir::ProtoJitDialect>();
  }
  void runOnOperation() final;

  // TODO: the same function may be called in multiple places, but only the
  // location of the first is saved. Can we store multiple locations in MLIR?
  // Should we set the location to something else, if this might be confusing?
  mlir::FuncOp getOrCreateStructEncodeFn(mlir::Location loc,
                                         OpBuilder::Listener* listener,
                                         StructType from, StructType to,
                                         PathAttr path);

  mlir::FuncOp getOrCreateStructDecodeFn(mlir::Location loc,
                                         OpBuilder::Listener* listener,
                                         StructType from, StructType to,
                                         ArrayAttr handlers);

  mlir::FuncOp getOrCreateVariantEncodeFn(mlir::Location loc,
                                          OpBuilder::Listener* listener,
                                          VariantType from, VariantType to,
                                          PathAttr path);

  mlir::FuncOp getOrCreateVariantDecodeFn(mlir::Location loc,
                                          OpBuilder::Listener* listener,
                                          VariantType from, VariantType to,
                                          ArrayAttr handlers);

  mlir::ModuleOp module() { return mlir::ModuleOp(getOperation()); }

 private:
  void transcodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                     VariantType src_type, VariantType dst_type,
                     const Term* src_term, const Term* dst_term, Value src,
                     Value dst, Value& buffer);

  mlir::FuncOp getOrCreateFn(mlir::Location loc, OpBuilder::Listener* listener,
                             llvm::StringRef prefix, const FnKey& key,
                             mlir::FunctionType signature);

  std::unordered_map<FnKey, Operation*> cached_fns;
  std::unordered_map<std::string, std::unordered_set<uint32_t>> used_names;
};

mlir::FuncOp GeneratePass::getOrCreateFn(mlir::Location loc,
                                         OpBuilder::Listener* listener,
                                         llvm::StringRef prefix,
                                         const FnKey& key,
                                         mlir::FunctionType signature) {
  if (auto it = cached_fns.find(key); it != cached_fns.end()) {
    return mlir::FuncOp{it->second};
  }

  std::string name;
  {
    // TODO: use twine for all these concats.
    llvm::raw_string_ostream os(name);

    // Start all internal functions with a '#', so we can identify
    // them for internal linkage later.
    os << "#" << prefix << "_";
    key.from.print(os);
    os << "_to_";
    key.to.print(os);
    os << "_attr_";
    printAttrForFunctionName(os, key.attr);

    // Types don't always print all the details involved in their uniquing.
    auto suffixes = used_names[name];
    uint32_t suffix = 0;
    while (suffixes.count(suffix)) {
      suffix = std::rand();
    }
    suffixes.insert(suffix);
    os << "_" << suffix;
  }

  // TODO: mark the function with some attribute (something like static?) so
  // LLVM knows it can throw away the body if all callsites are inlined.
  mlir::OpBuilder _ = mlir::OpBuilder::atBlockBegin(module().getBody());
  _.setListener(listener);
  return _.create<mlir::FuncOp>(loc, name, signature);
}

void GeneratePass::transcodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                                 VariantType src_type, VariantType dst_type,
                                 const Term* src_term, const Term* dst_term,
                                 Value src, Value dst, Value& buffer) {
  if (dst_term != nullptr) {
    auto src_body = _.create<ir2::ProjectOp>(loc, src_term->type, src,
                                             src_type.term_offset());
    auto dst_body = _.create<ir2::ProjectOp>(loc, dst_term->type, dst,
                                             dst_type.term_offset());

    if (dst_type.isa<OutlineVariantType>()) {
      // TODO: ensure sufficient space if decoding into an outline variant.
      buffer = _.create<ir2::ProjectOp>(
          loc, buffer.getType(), buffer,
          dst_term->type.cast<ValueType>().head_size());
    }

    _.create<ir2::TranscodeOp>(loc, buffer.getType(), src_body, dst_body,
                               buffer);
  }

  _.create<TagOp>(loc, dst,
                  dst_term == nullptr ? VariantType::kUndefTag : dst_term->tag);
}

mlir::FuncOp GeneratePass::getOrCreateStructEncodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, StructType from,
    StructType to, PathAttr path) {
  auto* ctx = &getContext();
  auto fntype =
      mlir::FunctionType::get(ctx, {from, to, types::RawBufferType::get(ctx)},
                              types::RawBufferType::get(ctx));
  auto key = FnKey{from, to, path};
  auto func = getOrCreateFn(loc, listener, "enc", key, fntype);

  if (!func.isDeclaration()) {
    return func;
  }

  auto entryBlock = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entryBlock);
  _.setListener(listener);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);

  // Encode each field in the target struct.
  // TODO: cache this map from names to fields in the source struct?
  llvm::StringMap<const StructField*> from_fields;
  for (auto& field : from->fields) {
    from_fields[field.name] = &field;
  }

  Value result_buf = func.getArgument(2);
  for (auto& to_field : to->fields) {
    // Project out the target field.
    auto dst_field =
        _.create<ir2::ProjectOp>(loc, to_field.type, dst, to_field.offset);
    if (auto it = from_fields.find(to_field.name); it != from_fields.end()) {
      auto& from_field = it->second;
      auto src_field = _.create<ir2::ProjectOp>(loc, from_field->type, src,
                                                from_field->offset);

      // If the target field exists in the source struct, encode
      // the source field into the target.
      result_buf = _.create<ir2::EncodeOp>(loc, RawBufferType::get(ctx),
                                           src_field, dst_field, result_buf,
                                           path.into(to_field.name));
    } else {
      // Otherwise fill in the target field with a default value.
      _.create<ir2::DefaultOp>(loc, dst_field);
    }
  }

  _.create<ReturnOp>(loc, result_buf);

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateStructDecodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, StructType from,
    StructType to, ArrayAttr handlers) {
  auto* ctx = &getContext();
  auto fntype =
      mlir::FunctionType::get(ctx, {from, to, BoundedBufferType::get(ctx)},
                              BoundedBufferType::get(ctx));

  auto key = FnKey{from, to, handlers};
  auto func = getOrCreateFn(loc, listener, "dec", key, fntype);

  if (!func.isDeclaration()) {
    return func;
  }

  auto entryBlock = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entryBlock);
  _.setListener(listener);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);
  Value result_buf = func.getArgument(2);

  // Decode each field into the target struct.
  // TODO: cache this map from names to fields in the source struct?
  llvm::StringMap<const StructField*> from_fields;
  for (auto& field : from->fields) {
    from_fields[field.name] = &field;
  }

  for (auto& to_field : to->fields) {
    // Project out the target field.
    auto dst_field =
        _.create<ir2::ProjectOp>(loc, to_field.type, dst, to_field.offset);
    if (auto it = from_fields.find(std::string{to_field.name});
        it != from_fields.end()) {
      auto& from_field = it->second;
      auto src_field = _.create<ir2::ProjectOp>(loc, from_field->type, src,
                                                from_field->offset);

      llvm::SmallVector<mlir::Attribute> field_handlers;

      for (auto& attr : handlers) {
        auto handler = attr.cast<DispatchHandlerAttr>();
        if (!handler.path().startsWith(to_field.name)) continue;
        field_handlers.emplace_back(DispatchHandlerAttr::get(
            ctx, std::make_pair(handler.path().into(to_field.name),
                                handler.address())));
      }

      // If the target field exists in the source struct, encode
      // the source field into the target.
      auto& ctx = getContext();
      result_buf = _.create<ir2::DecodeOp>(
          loc, BoundedBufferType::get(&ctx), src_field, dst_field, result_buf,
          mlir::ArrayAttr::get(&ctx, field_handlers));
    } else {
      // Otherwise fill in the target field with a default value.
      _.create<ir2::DefaultOp>(loc, dst_field);
    }
  }

  _.create<ReturnOp>(loc, result_buf);

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateVariantEncodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, VariantType src_type,
    VariantType dst_type, PathAttr path) {
  auto fntype = mlir::FunctionType::get(
      &getContext(),
      {src_type, dst_type, types::RawBufferType::get(&getContext())},
      types::RawBufferType::get(&getContext()));
  auto key = FnKey{src_type, dst_type, path};
  auto func = getOrCreateFn(loc, listener, "enc", key, fntype);

  if (!func.isDeclaration()) {
    return func;
  }

  auto entryBlock = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entryBlock);
  _.setListener(listener);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);

  std::map<std::string, const Term*> dst_terms;
  for (auto& term : dst_type.terms()) {
    dst_terms.emplace(term.name.str(), &term);
  }

  if (path.getValue().size() > 0) {
    // Source term is known. Encode it, or encode undef if missing in the
    // target.
    auto src_term_name = path.getValue()[0];

    const Term* src_term = nullptr;
    for (auto& term : src_type.terms()) {
      if (term.name == src_term_name) src_term = &term;
    }
    assert(src_term != nullptr);

    Value result_buf = func.getArgument(2);
    if (auto it = dst_terms.find(src_term_name.str()); it != dst_terms.end()) {
      transcodeTerm(_, loc, src_type, dst_type, src_term, it->second, src, dst,
                    result_buf);
    } else {
      transcodeTerm(_, loc, src_type, dst_type, src_term, nullptr, src, dst,
                    result_buf);
    }

    _.create<ReturnOp>(loc, result_buf);
  } else {
    // Source term is not known; dispatch on it.
    llvm::SmallVector<std::pair<intptr_t, mlir::Block*>, 4> blocks;

    // Check for undef in the source.
    {
      auto block = _.createBlock(&func.body());
      _.setInsertionPointToStart(block);
      Value result_buf = func.getArgument(2);

      transcodeTerm(_, loc, src_type, dst_type, nullptr, nullptr, src, dst,
                    result_buf);

      _.create<ReturnOp>(loc, result_buf);
      blocks.emplace_back(VariantType::kUndefTag, block);
    }

    for (auto& src_term : src_type.terms()) {
      auto block = _.createBlock(&func.body());
      _.setInsertionPointToStart(block);
      Value result_buf = func.getArgument(2);

      if (auto it = dst_terms.find(src_term.name.str());
          it != dst_terms.end()) {
        transcodeTerm(_, loc, src_type, dst_type, &src_term, it->second, src,
                      dst, result_buf);
      } else {
        transcodeTerm(_, loc, src_type, dst_type, &src_term, nullptr, src, dst,
                      result_buf);
      }

      _.create<ReturnOp>(loc, result_buf);

      blocks.emplace_back(src_term.tag, block);
    }

    std::sort(blocks.begin(), blocks.end());

    _.setInsertionPointToEnd(entryBlock);

    llvm::SmallVector<mlir::Block*, 4> succs;
    for (auto& [_, block] : blocks) {
      succs.push_back(block);
    }

    _.create<MatchOp>(loc, src, succs);
  }

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateVariantDecodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, VariantType src_type,
    VariantType dst_type, mlir::ArrayAttr handlers) {
  auto C = &getContext();

  auto fntype = mlir::FunctionType::get(
      C, {src_type, dst_type, types::BoundedBufferType::get(C)},
      types::BoundedBufferType::get(C));
  auto key = FnKey{src_type, dst_type, handlers};
  auto func = getOrCreateFn(loc, listener, "dec", key, fntype);

  if (!func.isDeclaration()) {
    return func;
  }

  auto entryBlock = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entryBlock);
  _.setListener(listener);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);

  std::map<std::string, const Term*> dst_terms;
  for (auto& term : dst_type.terms()) {
    dst_terms.emplace(term.name.str(), &term);
  }

  std::map<std::string, const void*> handler_map;
  for (auto& attr : handlers) {
    auto handler = attr.cast<DispatchHandlerAttr>();
    handler_map.emplace(handler.path().getValue()[0], handler.address());
  }

  llvm::SmallVector<std::pair<intptr_t, mlir::Block*>, 4> blocks;

  // Check for undef in the source.
  {
    auto block = _.createBlock(&func.body());
    _.setInsertionPointToStart(block);
    Value result_buf = func.getArgument(2);

    transcodeTerm(_, loc, src_type, dst_type, nullptr, nullptr, src, dst,
                  result_buf);

    // TODO: determine undef name
    if (auto it = handler_map.find("undef"); it != handler_map.end()) {
      _.create<SetCallbackOp>(
          loc, mlir::IntegerAttr::get(mlir::IndexType::get(C),
                                      reinterpret_cast<int64_t>(it->second)));
    }

    _.create<ReturnOp>(loc, result_buf);
    blocks.emplace_back(VariantType::kUndefTag, block);
  }

  for (auto& src_term : src_type.terms()) {
    auto block = _.createBlock(&func.body());
    _.setInsertionPointToStart(block);
    Value result_buf = func.getArgument(2);

    if (auto it = dst_terms.find(src_term.name.str()); it != dst_terms.end()) {
      transcodeTerm(_, loc, src_type, dst_type, &src_term, it->second, src, dst,
                    result_buf);
    } else {
      transcodeTerm(_, loc, src_type, dst_type, &src_term, nullptr, src, dst,
                    result_buf);
    }

    if (auto it = handler_map.find(src_term.name.str());
        it != handler_map.end()) {
      _.create<SetCallbackOp>(
          loc, mlir::IntegerAttr::get(mlir::IndexType::get(C),
                                      reinterpret_cast<int64_t>(it->second)));
    }

    _.create<ReturnOp>(loc, result_buf);

    blocks.emplace_back(src_term.tag, block);
  }

  std::sort(blocks.begin(), blocks.end());

  _.setInsertionPointToEnd(entryBlock);

  llvm::SmallVector<mlir::Block*, 4> succs;
  for (auto& [_, block] : blocks) {
    succs.push_back(block);
  }

  _.create<MatchOp>(loc, src, succs);

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

}  // namespace

struct EncodeFunctionLowering : public OpConversionPattern<EncodeFunctionOp> {
  using OpConversionPattern<EncodeFunctionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(EncodeFunctionOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult EncodeFunctionLowering::matchAndRewrite(
    EncodeFunctionOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  _.eraseOp(op);

  auto L = op.getLoc();
  auto C = _.getContext();

  const auto& proto = op.dst().cast<ProtocolType>();

  // Create a function with the given name, accepting parameters of the source
  // type and the protocol type. The function contains a single EncodeOp.
  auto func = mlir::FuncOp::create(
      L, op.name(),
      _.getFunctionType({op.src(), RawBufferType::get(C)}, llvm::None));

  mlir::ModuleOp module{op.getOperation()->getParentOp()};
  module.push_back(func);
  auto* entry_block = func.addEntryBlock();
  _.setInsertionPointToStart(entry_block);

  auto dst =
      _.create<ir2::ProjectOp>(L, proto->head, func.getArgument(1), Bits(0));

  auto dst_buf = _.create<ir2::ProjectOp>(
      L, RawBufferType::get(C), func.getArgument(1), proto->head.head_size());

  _.create<EncodeOp>(L, RawBufferType::get(C), func.getArgument(0), dst,
                     dst_buf, op.src_path());

  _.create<ReturnOp>(L);

  return success();
}

struct DecodeFunctionLowering : public OpConversionPattern<DecodeFunctionOp> {
  using OpConversionPattern<DecodeFunctionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(DecodeFunctionOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult DecodeFunctionLowering::matchAndRewrite(
    DecodeFunctionOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  _.eraseOp(op);

  auto loc = op.getLoc();
  auto ctx = _.getContext();

  const auto& proto = op.src().cast<ProtocolType>();

  // Create a function with the given name, accepting a protocol, destination
  // type, buffer, and user-state pointer. The function contains a single
  // DecodeOp.
  auto func = mlir::FuncOp::create(
      loc, op.name(),
      _.getFunctionType(
          {op.src(), op.dst(), types::BoundedBufferType::get(getContext()),
           UserStateType::get(getContext())},
          types::BoundedBufferType::get(getContext())));

  mlir::ModuleOp module{op.getOperation()->getParentOp()};
  module.push_back(func);
  auto* entry_block = func.addEntryBlock();
  _.setInsertionPointToStart(entry_block);

  // Create an exception handler to catch any decode errors.
  auto cth = _.create<DecodeCatchOp>(loc, types::BoundedBufferType::get(ctx));
  _.create<ReturnOp>(loc, ValueRange{cth});

  auto* catch_body = new Block();
  cth.body().push_back(catch_body);

  _.setInsertionPointToStart(catch_body);

  auto src =
      _.create<ir2::ProjectOp>(loc, proto->head, func.getArgument(0), Bits(0));

  Value buf = _.create<DecodeOp>(loc, BoundedBufferType::get(ctx), src,
                                 func.getArgument(1), func.getArgument(2),
                                 op.handlers());

  _.create<InvokeCallbackOp>(loc, func.getArgument(1), func.getArgument(3));

  _.create<YieldOp>(loc, buf);

  return success();
}

struct EncodeOpLowering : public OpConversionPattern<EncodeOp> {
  EncodeOpLowering(MLIRContext* C, GeneratePass* pass)
      : OpConversionPattern<EncodeOp>(C), pass(pass) {
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(EncodeOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  GeneratePass* pass;
};

LogicalResult EncodeOpLowering::matchAndRewrite(
    EncodeOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto L = op.getLoc();
  auto src_type = op.src().getType();
  auto dst_type = op.dst().getType();

  // Reduce Encode to Transcode for primitives.
  if (src_type.isa<IntType>() && dst_type.isa<IntType>()) {
    _.create<TranscodeOp>(L, /*resultType0=*/mlir::Type{}, op.src(), op.dst(),
                          /*buf=*/mlir::Value{});
    _.replaceOp(op, op.buf());
    return mlir::success();
  }

  // Call an outlined encoding routine for structs.
  if (src_type.isa<StructType>() && dst_type.isa<StructType>()) {
    auto fn = pass->getOrCreateStructEncodeFn(
        L, _.getListener(), src_type.cast<StructType>(),
        dst_type.cast<StructType>(), op.path());
    auto call = _.create<mlir::CallOp>(L, fn, operands);
    _.replaceOp(op, call.getResult(0));
    return success();
  }

  // Call an outlined encoding routine for variants.
  if (src_type.isa<VariantType>() && dst_type.isa<VariantType>()) {
    auto fn = pass->getOrCreateVariantEncodeFn(
        L, _.getListener(), src_type.cast<VariantType>(),
        dst_type.cast<VariantType>(), op.path());
    auto call = _.create<mlir::CallOp>(L, fn, operands);
    _.replaceOp(op, call.getResult(0));
    return success();
  }

  return failure();
}

struct DecodeOpLowering : public OpConversionPattern<DecodeOp> {
  DecodeOpLowering(MLIRContext* C, GeneratePass* pass)
      : OpConversionPattern<DecodeOp>(C), pass(pass) {
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(DecodeOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  GeneratePass* pass;
};

LogicalResult DecodeOpLowering::matchAndRewrite(
    DecodeOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto L = op.getLoc();
  auto src_type = op.src().getType();
  auto dst_type = op.dst().getType();

  // Reduce Decode to Transcode for primitives.
  if (src_type.isa<IntType>() && dst_type.isa<IntType>()) {
    _.create<TranscodeOp>(L, /*resultType0=*/mlir::Type{}, op.src(), op.dst(),
                          /*buf=*/mlir::Value{});
    _.replaceOp(op, op.buf());
    return success();
  }

  if (src_type.isa<StructType>() && dst_type.isa<StructType>()) {
    auto fn = pass->getOrCreateStructDecodeFn(
        L, _.getListener(), src_type.cast<StructType>(),
        dst_type.cast<StructType>(), op.handlers());
    auto call = _.create<mlir::CallOp>(L, fn, operands);
    _.replaceOp(op, call.getResult(0));
    return success();
  }

  if (src_type.isa<VariantType>() && dst_type.isa<VariantType>()) {
    auto fn = pass->getOrCreateVariantDecodeFn(
        L, _.getListener(), src_type.cast<VariantType>(),
        dst_type.cast<VariantType>(), op.handlers());
    auto call = _.create<mlir::CallOp>(L, fn, operands);
    _.replaceOp(op, call.getResult(0));
    return success();
  }

  return failure();
}

void GeneratePass::runOnOperation() {
  auto* ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addLegalDialect<StandardOpsDialect, ProtoJitDialect>();
  target.addLegalOp<FuncOp>();
  target.addIllegalOp<EncodeFunctionOp, DecodeFunctionOp, SizeFunctionOp,
                      EncodeOp, DecodeOp>();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<EncodeFunctionLowering, DecodeFunctionLowering>(ctx)
      .insert<EncodeOpLowering, DecodeOpLowering>(ctx, this);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createIRGenPass() {
  return std::make_unique<GeneratePass>();
}

}  // namespace pj
