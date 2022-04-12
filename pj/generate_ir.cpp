#include <unordered_map>
#include <unordered_set>

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "defer.hpp"
#include "ir.hpp"
#include "util.hpp"

namespace pj {
struct FnKey {
  types::ValueType from;
  types::ValueType to;
  types::PathAttr path;
  mlir::ArrayAttr handlers;
  mlir::Type buf_type;

  bool operator==(const FnKey& other) const {
    return from == other.from && to == other.to && path == other.path &&
           handlers == other.handlers && buf_type == other.buf_type;
  }
};
}  // namespace pj

namespace std {
template <>
struct hash<pj::FnKey> {
  size_t operator()(const pj::FnKey& key) const {
    using ::llvm::hash_value;
    return llvm::hash_combine(hash_value(key.from), hash_value(key.to),
                              hash_value(key.path), hash_value(key.handlers),
                              hash_value(key.buf_type));
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
  mlir::FuncOp getOrCreateStructTranscodeFn(mlir::Location loc,
                                            OpBuilder::Listener* listener,
                                            StructType from, StructType to,
                                            PathAttr path, ArrayAttr handlers,
                                            mlir::Type buf_type);

  mlir::FuncOp getOrCreateVariantTranscodeFn(mlir::Location loc,
                                             OpBuilder::Listener* listener,
                                             VariantType from, VariantType to,
                                             PathAttr path, ArrayAttr handlers,
                                             mlir::Type buf_type);

  mlir::FuncOp getOrCreateArrayTranscodeFn(mlir::Location loc,
                                           OpBuilder::Listener* listener,
                                           ArrayType from, ArrayType to,
                                           mlir::Type buf_type);

  mlir::ModuleOp module() { return mlir::ModuleOp(getOperation()); }

 private:
  void transcodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                     VariantType src_type, VariantType dst_type,
                     const Term* src_term, const Term* dst_term, Value src,
                     Value dst, Value& buffer,
                     llvm::StringMap<const void*>& handler_map);

  mlir::FuncOp getOrCreateFn(mlir::Location loc, OpBuilder::Listener* listener,
                             llvm::StringRef prefix, const FnKey& key);

  std::unordered_map<FnKey, Operation*> cached_fns;
  std::unordered_map<std::string, std::unordered_set<uint32_t>> used_names;
};

mlir::FuncOp GeneratePass::getOrCreateFn(mlir::Location loc,
                                         OpBuilder::Listener* listener,
                                         llvm::StringRef prefix,
                                         const FnKey& key) {
  if (auto it = cached_fns.find(key); it != cached_fns.end()) {
    return mlir::FuncOp{it->second};
  }

  auto* ctx = &getContext();
  auto fntype = mlir::FunctionType::get(ctx, {key.from, key.to, key.buf_type},
                                        key.buf_type);

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
    os << "_path_";
    printAttrForFunctionName(os, key.path);
    os << "_handlers_";
    printAttrForFunctionName(os, key.handlers);
    if (key.buf_type.isa<BoundedBufferType>()) {
      os << "_checked";
    }

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

  auto func = _.create<mlir::FuncOp>(loc, name, fntype);

  for (size_t i = 0; i < func.getNumArguments(); ++i) {
    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(&getContext()));
  }

  return func;
}

void GeneratePass::transcodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                                 VariantType src_type, VariantType dst_type,
                                 const Term* src_term, const Term* dst_term,
                                 Value src, Value dst, Value& buffer,
                                 llvm::StringMap<const void*>& handler_map) {
  auto* ctx = _.getContext();

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
                               buffer, PathAttr::none(ctx),
                               ArrayAttr::get(ctx, {}));
  }

  _.create<TagOp>(loc, dst,
                  dst_term == nullptr ? VariantType::kUndefTag : dst_term->tag);

  if (auto it = handler_map.find(dst_term ? dst_term->name : "undef");
      it != handler_map.end()) {
    _.create<SetCallbackOp>(
        loc, mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                    reinterpret_cast<int64_t>(it->second)));
  }
}

mlir::FuncOp GeneratePass::getOrCreateStructTranscodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, StructType from,
    StructType to, PathAttr path, ArrayAttr handlers, mlir::Type buf_type) {
  auto* ctx = &getContext();
  auto key = FnKey{from, to, path, handlers, buf_type};
  auto func = getOrCreateFn(loc, listener, "xcd", key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
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
    llvm::SmallVector<mlir::Attribute> field_handlers;

    for (auto& attr : handlers) {
      auto handler = attr.cast<DispatchHandlerAttr>();
      if (!handler.path().startsWith(to_field.name)) continue;
      field_handlers.emplace_back(DispatchHandlerAttr::get(
          ctx, std::make_pair(handler.path().into(to_field.name),
                              handler.address())));
    }

    auto handlers_attr = mlir::ArrayAttr::get(ctx, field_handlers);

    // Project out the target field.
    auto dst_field =
        _.create<ir2::ProjectOp>(loc, to_field.type, dst, to_field.offset);
    if (auto it = from_fields.find(to_field.name); it != from_fields.end()) {
      auto& from_field = it->second;
      auto src_field = _.create<ir2::ProjectOp>(loc, from_field->type, src,
                                                from_field->offset);

      // If the target field exists in the source struct, encode
      // the source field into the target.
      result_buf = _.create<ir2::TranscodeOp>(
          loc, result_buf.getType(), src_field, dst_field, result_buf,
          path.into(to_field.name), handlers_attr);
    } else {
      // Otherwise fill in the target field with a default value.
      _.create<ir2::DefaultOp>(loc, dst_field, handlers_attr);
    }
  }

  _.create<ReturnOp>(loc, result_buf);

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateVariantTranscodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, VariantType src_type,
    VariantType dst_type, PathAttr path, ArrayAttr handlers,
    mlir::Type buf_type) {
  auto key = FnKey{src_type, dst_type, path, handlers, buf_type};
  auto func = getOrCreateFn(loc, listener, "enc", key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
  _.setListener(listener);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);

  std::map<std::string, const Term*> dst_terms;
  for (auto& term : dst_type.terms()) {
    dst_terms.emplace(term.name.str(), &term);
  }

  llvm::StringMap<const void*> handler_map;
  for (auto& attr : handlers) {
    auto handler = attr.cast<DispatchHandlerAttr>();
    handler_map.insert({handler.path().getValue()[0], handler.address()});
  }

  llvm::SmallVector<std::pair<intptr_t, mlir::Block*>, 4> blocks;

  if (path.getValue().size() > 0) {
    // Source term is known. Encode it, or encode undef if missing in the
    // target.
    auto src_term_name = path.getValue()[0];

    const Term* src_term = nullptr;
    for (auto& term : src_type.terms()) {
      if (term.name == src_term_name) src_term = &term;
    }
    assert(src_term_name == "undef" || src_term != nullptr);

    Value result_buf = func.getArgument(2);
    if (auto it = dst_terms.find(src_term_name.str()); it != dst_terms.end()) {
      assert(src_term_name != "undef");
      transcodeTerm(_, loc, src_type, dst_type, src_term, it->second, src, dst,
                    result_buf, handler_map);
    } else {
      transcodeTerm(_, loc, src_type, dst_type, src_term, nullptr, src, dst,
                    result_buf, handler_map);
    }

    _.create<ReturnOp>(loc, result_buf);
  } else {
    // Source term is not known; dispatch on it.
    llvm::SmallVector<std::pair<intptr_t, mlir::Block*>, 4> blocks;

    // Check for undef in the source.
    {
      auto* block = _.createBlock(&func.body());
      _.setInsertionPointToStart(block);
      Value result_buf = func.getArgument(2);

      transcodeTerm(_, loc, src_type, dst_type, nullptr, nullptr, src, dst,
                    result_buf, handler_map);

      _.create<ReturnOp>(loc, result_buf);
      blocks.emplace_back(VariantType::kUndefTag, block);
    }

    for (auto& src_term : src_type.terms()) {
      auto* block = _.createBlock(&func.body());
      _.setInsertionPointToStart(block);
      Value result_buf = func.getArgument(2);

      if (auto it = dst_terms.find(src_term.name.str());
          it != dst_terms.end()) {
        transcodeTerm(_, loc, src_type, dst_type, &src_term, it->second, src,
                      dst, result_buf, handler_map);
      } else {
        transcodeTerm(_, loc, src_type, dst_type, &src_term, nullptr, src, dst,
                      result_buf, handler_map);
      }

      _.create<ReturnOp>(loc, result_buf);

      blocks.emplace_back(src_term.tag, block);
    }

    std::sort(blocks.begin(), blocks.end());

    _.setInsertionPointToEnd(entry_block);

    llvm::SmallVector<mlir::Block*, 4> succs;
    for (auto& [_, block] : blocks) {
      succs.push_back(block);
    }

    _.create<MatchOp>(loc, src, succs);
  }

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateArrayTranscodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, ArrayType from,
    ArrayType to, mlir::Type buf_type) {
  auto* ctx = &getContext();
  auto key =
      FnKey{from, to, PathAttr::none(ctx), ArrayAttr::get(ctx, {}), buf_type};
  auto func = getOrCreateFn(loc, listener, "xcd", key);

  for (size_t i : {0, 1, 2}) {
    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(ctx));
  }

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
  _.setListener(listener);

  Value src = func.getArgument(0), dst = func.getArgument(1),
        result_buf = func.getArgument(2);

  auto transcode_len = std::min(from->length, to->length);

  auto loop_start =
      _.create<ConstantOp>(loc, _.getIntegerAttr(_.getIndexType(), 0));
  auto loop_end = _.create<ConstantOp>(
      loc, _.getIntegerAttr(_.getIndexType(), transcode_len));
  auto step = _.create<ConstantOp>(loc, _.getIntegerAttr(_.getIndexType(), 1));

  auto transcode_loop = _.create<scf::ForOp>(loc, loop_start, loop_end, step,
                                             ValueRange{result_buf});

  loop_start = loop_end;
  loop_end =
      _.create<ConstantOp>(loc, _.getIntegerAttr(_.getIndexType(), to->length));
  auto default_loop =
      _.create<scf::ForOp>(loc, loop_start, loop_end, step, ValueRange{});

  _.create<ReturnOp>(loc, transcode_loop.getResult(0));

  // Transcode available elements.
  {
    auto* body = transcode_loop.getBody();
    _.setInsertionPointToEnd(body);

    Value idx = body->getArgument(0);
    result_buf = body->getArgument(1);

    auto src_elem = _.create<IndexOp>(loc, from->elem, src, idx);
    auto dst_elem = _.create<IndexOp>(loc, to->elem, dst, idx);

    result_buf = _.create<TranscodeOp>(
        loc, result_buf.getType(), src_elem, dst_elem, result_buf,
        PathAttr::none(ctx), ArrayAttr::get(ctx, {}));

    _.create<scf::YieldOp>(loc, result_buf);
  }

  // Fill the rest with defaults.
  {
    auto* body = default_loop.getBody();
    _.setInsertionPointToStart(body);
    Value idx = body->getArgument(0);
    auto dst_elem = _.create<IndexOp>(loc, to->elem, dst, idx);
    _.create<DefaultOp>(loc, dst_elem, ArrayAttr::get(ctx, {}));
    // ForOp::build automatically creates a terminating yield since
    // we have no loop carried variables.
  }

  cached_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

struct EncodeFunctionLowering : public OpConversionPattern<EncodeFunctionOp> {
  using OpConversionPattern<EncodeFunctionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(EncodeFunctionOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult EncodeFunctionLowering::matchAndRewrite(
    EncodeFunctionOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  _.eraseOp(op);

  auto loc = op.getLoc();
  auto* ctx = _.getContext();

  const auto& proto = op.dst().cast<ProtocolType>();

  // Create a function with the given name, accepting parameters of the source
  // type and the protocol type. The function contains a single EncodeOp.
  auto func = mlir::FuncOp::create(
      loc, op.name(),
      _.getFunctionType({op.src(), RawBufferType::get(ctx)}, llvm::None));

  for (size_t i : {0, 1}) {
    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(ctx));
  }

  mlir::ModuleOp module{op.getOperation()->getParentOp()};
  module.push_back(func);
  auto* entry_block = func.addEntryBlock();
  _.setInsertionPointToStart(entry_block);

  auto dst =
      _.create<ir2::ProjectOp>(loc, proto->head, func.getArgument(1), Bits(0));

  auto dst_buf =
      _.create<ir2::ProjectOp>(loc, RawBufferType::get(ctx),
                               func.getArgument(1), proto->head.head_size());

  _.create<TranscodeOp>(loc, RawBufferType::get(ctx), func.getArgument(0), dst,
                        dst_buf, op.src_path(), ArrayAttr::get(ctx, {}));

  _.create<ReturnOp>(loc);

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
  auto* ctx = _.getContext();

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

  for (size_t i : {0, 1, 2}) {
    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(ctx));
  }

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

  Value buf = _.create<TranscodeOp>(loc, BoundedBufferType::get(ctx), src,
                                    func.getArgument(1), func.getArgument(2),
                                    PathAttr::none(ctx), op.handlers());

  _.create<InvokeCallbackOp>(loc, func.getArgument(1), func.getArgument(3));

  _.create<YieldOp>(loc, buf);

  return success();
}

struct TranscodeOpLowering : public OpConversionPattern<TranscodeOp> {
  TranscodeOpLowering(MLIRContext* ctx, GeneratePass* pass)
      : OpConversionPattern<TranscodeOp>(ctx), pass(pass) {
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(TranscodeOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  GeneratePass* pass;
};

LogicalResult TranscodeOpLowering::matchAndRewrite(
    TranscodeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto src_type = op.src().getType();
  auto dst_type = op.dst().getType();

  FuncOp fn;

  if (src_type.isa<StructType>() && dst_type.isa<StructType>()) {
    fn = pass->getOrCreateStructTranscodeFn(
        loc, _.getListener(), src_type.cast<StructType>(),
        dst_type.cast<StructType>(), op.path(), op.handlers(),
        op.getResult().getType());
  } else if (src_type.isa<VariantType>() && dst_type.isa<VariantType>()) {
    fn = pass->getOrCreateVariantTranscodeFn(
        loc, _.getListener(), src_type.cast<VariantType>(),
        dst_type.cast<VariantType>(), op.path(), op.handlers(),
        op.getResult().getType());
  } else if (src_type.isa<ArrayType>() && dst_type.isa<ArrayType>()) {
    fn = pass->getOrCreateArrayTranscodeFn(
        loc, _.getListener(), src_type.cast<ArrayType>(),
        dst_type.cast<ArrayType>(), op.getResult().getType());
  } else {
    return failure();
  }

  auto call = _.create<mlir::CallOp>(loc, fn, operands);
  _.replaceOp(op, call.getResult(0));
  return success();
}

}  // namespace

void GeneratePass::runOnOperation() {
  auto* ctx = &getContext();

  ConversionTarget target(*ctx);
  target
      .addLegalDialect<StandardOpsDialect, scf::SCFDialect, ProtoJitDialect>();
  target.addLegalOp<FuncOp>();
  target.addIllegalOp<EncodeFunctionOp, DecodeFunctionOp, SizeFunctionOp>();
  target.addDynamicallyLegalOp<TranscodeOp>([](TranscodeOp op) {
    return op.src().getType().isa<IntType>() &&
           op.dst().getType().isa<IntType>();
  });

  OwningRewritePatternList patterns(ctx);
  patterns.insert<EncodeFunctionLowering, DecodeFunctionLowering>(ctx)
      .insert<TranscodeOpLowering>(ctx, this);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createIRGenPass() {
  return std::make_unique<GeneratePass>();
}

}  // namespace pj
