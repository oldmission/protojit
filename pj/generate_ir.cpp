#include <unordered_map>
#include <unordered_set>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Pass/Pass.h>

#include <pj/util.hpp>

#include "defer.hpp"
#include "ir.hpp"
#include "reflect.hpp"

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

  mlir::FuncOp getOrCreateVectorTranscodeFn(mlir::Location loc,
                                            OpBuilder::Listener* listener,
                                            VectorType from, VectorType to,
                                            mlir::Type buf_type);

  mlir::FuncOp getOrCreateReflectFn(mlir::Location loc,
                                    OpBuilder::Listener* listener,
                                    ValueType from, AnyType to,
                                    mlir::Type buf_type);

  mlir::ModuleOp module() { return mlir::ModuleOp(getOperation()); }

  mlir::Value buildIndex(mlir::Location loc, mlir::OpBuilder& _, size_t value) {
    return _.create<ConstantOp>(loc, _.getIntegerAttr(_.getIndexType(), value));
  }

  mlir::Value buildBool(mlir::Location loc, mlir::OpBuilder& _, bool value) {
    return _.create<ConstantOp>(loc, _.getBoolAttr(value));
  }

  Value projectPath(mlir::OpBuilder& _, mlir::Location loc, Value base,
                    PathAttr path_attr, bool frozen);

  Value generateTermCondition(mlir::OpBuilder& _, mlir::Location loc, Value src,
                              ArrayRef<TermAttribute> dst_attributes);

  void transcodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                     VariantType src_type, VariantType dst_type,
                     ValueType src_term_type,
                     llvm::SmallVector<const Term*, 1> dst_terms, Value src,
                     Value dst, Value& buffer,
                     llvm::StringMap<uint64_t>& handler_map);

  mlir::Value generateVectorCopyLoop(mlir::Location loc, mlir::OpBuilder& _,
                                     Value src, Value src_start,
                                     VectorRegion src_region, Value dst,
                                     Value dst_start, VectorRegion dst_region,
                                     Value copy_length, Value outline_buf,
                                     Value buf);

  mlir::Value transcodeInlineVector(mlir::Location loc, mlir::OpBuilder& _,
                                    Value src, Value dst, Value copy_length,
                                    Value buf);

  mlir::Value transcodeOutlineVector(mlir::Location loc, mlir::OpBuilder& _,
                                     Value src, Value dst, Value copy_length,
                                     Value buf);

  mlir::FuncOp getOrCreateFn(mlir::Location loc, OpBuilder::Listener* listener,
                             llvm::StringRef prefix, const FnKey& key);

  std::unordered_map<FnKey, Operation*> cached_fns_;
  std::unordered_map<std::string, std::unordered_set<uint32_t>> used_names_;
};

mlir::FuncOp GeneratePass::getOrCreateFn(mlir::Location loc,
                                         OpBuilder::Listener* listener,
                                         llvm::StringRef prefix,
                                         const FnKey& key) {
  if (auto it = cached_fns_.find(key); it != cached_fns_.end()) {
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
    } else if (key.buf_type.isa<DummyBufferType>()) {
      os << "_size";
    }

    // Types don't always print all the details involved in their uniquing.
    auto& suffixes = used_names_[name];
    uint32_t suffix = 0;
    while (suffixes.count(suffix)) {
      suffix = std::rand();
    }
    suffixes.insert(suffix);
    os << "_" << suffix;
  }

  ASSERT(!module().lookupSymbol(name));

  // TODO: mark the function with some attribute (something like static?) so
  // LLVM knows it can throw away the body if all callsites are inlined.
  mlir::OpBuilder _ = mlir::OpBuilder::atBlockBegin(module().getBody());
  _.setListener(listener);

  auto func = _.create<mlir::FuncOp>(loc, name, fntype);

  for (size_t i = 0; i < 2; ++i) {
    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(&getContext()));
  }
  if (!key.buf_type.isa<DummyBufferType>()) {
    func.setArgAttr(2, LLVM::LLVMDialect::getNoAliasAttrName(),
                    UnitAttr::get(&getContext()));
  }

  return func;
}

Value GeneratePass::projectPath(mlir::OpBuilder& _, mlir::Location loc,
                                Value base, PathAttr path_attr, bool frozen) {
  auto path = path_attr.getValue();
  if (path.size() == 1) {
    return base;
  }
  // If there are remaining fields in path, it must be a struct type.
  auto type = base.getType().cast<StructType>();
  auto it =
      std::find_if(type->fields.begin(), type->fields.end(),
                   [&](const StructField& f) { return f.name == path[0]; });
  assert(it != type->fields.end());
  auto body = _.create<ProjectOp>(loc, it->type, base, it->offset, frozen);
  return projectPath(_, loc, body, path_attr.narrow(), frozen);
}

Value GeneratePass::generateTermCondition(
    mlir::OpBuilder& _, mlir::Location loc, Value src,
    ArrayRef<TermAttribute> dst_attributes) {
  auto cond = buildBool(loc, _, true);
  for (const TermAttribute& attr : dst_attributes) {
    auto handle_undef = [&](const TermAttribute::Undef& undef) -> Value {
      return buildBool(loc, _, undef.is_default);
    };

    auto handle_vector_split =
        [&](const TermAttribute::VectorSplit& vs) -> Value {
      auto vec = projectPath(_, loc, src, vs.path, true);
      auto length = _.create<LengthOp>(loc, _.getIndexType(), vec);
      return _.create<CmpIOp>(loc,
                              vs.type == TermAttribute::VectorSplit::kInline
                                  ? CmpIPredicate::ule
                                  : CmpIPredicate::ugt,
                              length, buildIndex(loc, _, vs.inline_length));
    };

    auto cur_cond =
        std::visit(overloaded{handle_undef, handle_vector_split}, attr.value);
    cond = _.create<AndOp>(loc, cond, cur_cond);
  }
  return cond;
}

void GeneratePass::transcodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                                 VariantType src_type, VariantType dst_type,
                                 ValueType src_term_type,
                                 llvm::SmallVector<const Term*, 1> dst_terms,
                                 Value src, Value dst, Value& buffer,
                                 llvm::StringMap<uint64_t>& handler_map) {
  auto* ctx = _.getContext();

  const Term& default_term = dst_type.terms()[dst_type.default_term()];

  if (!dst_terms.empty()) {
    auto src_body = _.create<ir::ProjectOp>(loc, src_term_type, src,
                                            src_type.term_offset(), true);

    mlir::OpBuilder::InsertPoint ip;
    Value start_buf = buffer;
    for (uintptr_t i = 0; i < dst_terms.size(); ++i) {
      const Term* dst_term = dst_terms[i];
      auto cond = generateTermCondition(_, loc, src_body, dst_term->attributes);
      auto if_op = _.create<scf::IfOp>(loc, start_buf.getType(), cond,
                                       /*withElseRegion=*/true);
      if (i == 0) {
        buffer = if_op.getResult(0);
        ip = _.saveInsertionPoint();
      } else {
        _.create<scf::YieldOp>(loc, if_op.getResult(0));
      }

      {  // Fill in the body with the transcode to this specific dst term.
        _.setInsertionPointToStart(if_op.thenBlock());
        _.create<TagOp>(loc, dst, dst_term->tag);

        auto dst_body = _.create<ProjectOp>(loc, dst_term->type, dst,
                                            dst_type.term_offset());

        auto cur_buf = start_buf;
        if (dst_type.isa<OutlineVariantType>()) {
          cur_buf = _.create<AllocateOp>(
              loc, cur_buf.getType(), cur_buf,
              buildIndex(loc, _,
                         dst_term->type.cast<ValueType>().headSize().bytes()));
        }
        cur_buf = _.create<TranscodeOp>(loc, cur_buf.getType(), src_body,
                                        dst_body, cur_buf, PathAttr::none(ctx),
                                        ArrayAttr::get(ctx, {}));
        _.create<scf::YieldOp>(loc, cur_buf);
      }

      _.setInsertionPointToStart(if_op.elseBlock());
    }

    // The insertion point is inside the final else block.
    _.create<scf::YieldOp>(loc, start_buf);
    // TODO: add a debug assert that this else block is never reached

    _.restoreInsertionPoint(ip);
  } else {
    _.create<TagOp>(loc, dst, default_term.tag);
  }

  if (auto it = handler_map.find(dst_terms.empty() ? default_term.name
                                                   : dst_terms[0]->name);
      it != handler_map.end()) {
    assert(dst_terms.size() <= 1);
    _.create<SetCallbackOp>(
        loc, mlir::IntegerAttr::get(mlir::IndexType::get(ctx), it->second));
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

  // Make sure the field containing the outline variant corresponding to the
  // tag path (if there is one) gets transcoded first, because its data is
  // expected to come first in the buffer
  llvm::SmallVector<const StructField*, 4> to_fields;
  const StructField* outline_field = nullptr;
  for (auto& field : to->fields) {
    if (field.type.isa<OutlineVariantType>() ||
        (field.type.isa<StructType>() &&
         field.type.cast<StructType>()->outline_variant)) {
      outline_field = &field;
    } else {
      to_fields.emplace_back(&field);
    }
  }
  if (outline_field != nullptr) {
    to_fields.insert(to_fields.begin(), outline_field);
  }

  // Poison the entire area of the destination struct so that future
  // optimizations can overwrite the padding bytes.
  _.create<PoisonOp>(loc, dst, Bytes(0), to.headSize());

  Value result_buf = func.getArgument(2);
  for (const auto* to_field : to_fields) {
    llvm::SmallVector<mlir::Attribute> field_handlers;

    for (auto& attr : handlers) {
      auto handler = attr.cast<DispatchHandlerAttr>();
      if (!handler.path().startsWith(to_field->name)) continue;
      field_handlers.emplace_back(DispatchHandlerAttr::get(
          ctx, std::make_pair(handler.path().into(to_field->name),
                              handler.index())));
    }

    auto handlers_attr = mlir::ArrayAttr::get(ctx, field_handlers);

    // Project out the target field.
    auto dst_field =
        _.create<ir::ProjectOp>(loc, to_field->type, dst, to_field->offset);
    if (auto it = from_fields.find(to_field->name); it != from_fields.end()) {
      auto& from_field = it->second;
      auto src_field = _.create<ir::ProjectOp>(loc, from_field->type, src,
                                               from_field->offset, true);

      // If the target field exists in the source struct, encode
      // the source field into the target.
      result_buf = _.create<ir::TranscodeOp>(
          loc, result_buf.getType(), src_field, dst_field, result_buf,
          path.into(to_field->name), handlers_attr);
    } else {
      // Otherwise fill in the target field with a default value.
      _.create<ir::DefaultOp>(loc, result_buf.getType(), dst_field, result_buf,
                              handlers_attr);
    }
  }

  _.create<ReturnOp>(loc, result_buf);

  cached_fns_.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateVariantTranscodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, VariantType src_type,
    VariantType dst_type, PathAttr path, ArrayAttr handlers,
    mlir::Type buf_type) {
  auto key = FnKey{src_type, dst_type, path, handlers, buf_type};
  auto func = getOrCreateFn(loc, listener, "xcd", key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
  _.setListener(listener);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);

  llvm::StringMap<llvm::SmallVector<const Term*, 1>> dst_terms;
  for (auto& term : dst_type.terms()) {
    auto [it, _] = dst_terms.try_emplace(term.name);
    it->second.emplace_back(&term);
  }

  const Term* default_term = &dst_type.terms()[dst_type.default_term()];

  llvm::StringMap<uint64_t> handler_map;
  for (auto& attr : handlers) {
    auto handler = attr.cast<DispatchHandlerAttr>();
    handler_map.insert({handler.path().getValue()[0], handler.index()});
  }

  llvm::SmallVector<std::pair<intptr_t, mlir::Block*>, 4> blocks;

  if (path.getValue().size() > 0) {
    // Source term is known, so encode it immediately.
    auto src_term_name = path.getValue()[0];

    const Term* src_term =
        std::find_if(src_type.terms().begin(), src_type.terms().end(),
                     [&](const Term& t) { return t.name == src_term_name; });
    assert(src_term != nullptr);

    Value result_buf = func.getArgument(2);
    if (auto it = dst_terms.find(src_term_name.str()); it != dst_terms.end()) {
      transcodeTerm(_, loc, src_type, dst_type, src_term->type, it->second, src,
                    dst, result_buf, handler_map);

    } else {
      transcodeTerm(_, loc, src_type, dst_type, src_term->type, {default_term},
                    src, dst, result_buf, handler_map);
    }

    _.create<ReturnOp>(loc, result_buf);
  } else {
    // Source term is not known; dispatch on it.
    llvm::SmallVector<std::pair<intptr_t, mlir::Block*>, 4> blocks;

    auto* default_block = _.createBlock(&func.body());
    {
      _.setInsertionPointToStart(default_block);

      Value result_buf = func.getArgument(2);
      transcodeTerm(_, loc, src_type, dst_type, UnitType::get(_.getContext()),
                    {default_term}, src, dst, result_buf, handler_map);

      _.create<ReturnOp>(loc, result_buf);
    }

    for (const auto& src_term : src_type.terms()) {
      auto* block = _.createBlock(&func.body());
      _.setInsertionPointToStart(block);

      Value result_buf = func.getArgument(2);
      if (auto it = dst_terms.find(src_term.name.str());
          it != dst_terms.end()) {
        transcodeTerm(_, loc, src_type, dst_type, src_term.type, it->second,
                      src, dst, result_buf, handler_map);
      } else {
        transcodeTerm(_, loc, src_type, dst_type, src_term.type, {default_term},
                      src, dst, result_buf, handler_map);
      }

      _.create<ReturnOp>(loc, result_buf);

      blocks.emplace_back(src_term.tag, block);
    }

    std::sort(blocks.begin(), blocks.end());

    _.setInsertionPointToEnd(entry_block);

    llvm::SmallVector<mlir::Block*, 4> cases;
    for (auto& [_, block] : blocks) {
      cases.push_back(block);
    }

    _.create<MatchOp>(loc, src, default_block, cases);
  }

  cached_fns_.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateArrayTranscodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, ArrayType from,
    ArrayType to, mlir::Type buf_type) {
  auto* ctx = &getContext();
  auto key =
      FnKey{from, to, PathAttr::none(ctx), ArrayAttr::get(ctx, {}), buf_type};
  auto func = getOrCreateFn(loc, listener, "xcd", key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
  _.setListener(listener);

  Value src = func.getArgument(0), dst = func.getArgument(1),
        result_buf = func.getArgument(2);

  auto transcode_len = std::min(from->length, to->length);

  auto loop_start = buildIndex(loc, _, 0);
  auto loop_end = buildIndex(loc, _, transcode_len);
  auto step = buildIndex(loc, _, 1);

  auto transcode_loop = _.create<scf::ForOp>(loc, loop_start, loop_end, step,
                                             ValueRange{result_buf});

  loop_start = loop_end;
  loop_end = buildIndex(loc, _, to->length);
  auto default_loop =
      _.create<scf::ForOp>(loc, loop_start, loop_end, step, ValueRange{});

  auto final_buf = transcode_loop.getResult(0);
  _.create<ReturnOp>(loc, transcode_loop.getResult(0));

  // Transcode available elements.
  {
    auto* body = transcode_loop.getBody();
    _.setInsertionPointToEnd(body);

    Value idx = body->getArgument(0);
    result_buf = body->getArgument(1);

    auto src_elem = _.create<ArrayIndexOp>(loc, from->elem, src, idx);
    auto dst_elem = _.create<ArrayIndexOp>(loc, to->elem, dst, idx);

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
    auto dst_elem = _.create<ArrayIndexOp>(loc, to->elem, dst, idx);
    _.create<DefaultOp>(loc, final_buf.getType(), dst_elem, final_buf,
                        ArrayAttr::get(ctx, {}));
    // ForOp::build automatically creates a terminating yield since
    // we have no loop carried variables.
  }

  cached_fns_.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::Value GeneratePass::generateVectorCopyLoop(
    mlir::Location loc, mlir::OpBuilder& _, Value src, Value src_start,
    VectorRegion src_region, Value dst, Value dst_start,
    VectorRegion dst_region, Value copy_length, Value outline_buf, Value buf) {
  auto* ctx = _.getContext();
  auto loop_start = buildIndex(loc, _, 0);
  auto loop_end = copy_length;
  auto step = buildIndex(loc, _, 1);
  auto copy_loop =
      _.create<scf::ForOp>(loc, loop_start, loop_end, step, ValueRange{buf});
  auto result_buf = copy_loop.getResult(0);
  auto ip = _.saveInsertionPoint();

  auto* body = copy_loop.getBody();
  _.setInsertionPointToEnd(body);

  Value idx = body->getArgument(0);
  Value loop_buf = body->getArgument(1);

  Value src_idx = _.create<AddIOp>(loc, src_start, idx);
  Value dst_idx = _.create<AddIOp>(loc, dst_start, idx);

  auto src_elem =
      _.create<VectorIndexOp>(loc, src.getType().cast<VectorType>()->elem, src,
                              src_idx, src_region, outline_buf);
  auto dst_elem =
      _.create<VectorIndexOp>(loc, dst.getType().cast<VectorType>()->elem, dst,
                              dst_idx, dst_region, outline_buf);

  loop_buf = _.create<TranscodeOp>(loc, loop_buf.getType(), src_elem, dst_elem,
                                   loop_buf, PathAttr::none(ctx),
                                   ArrayAttr::get(ctx, {}));
  _.create<scf::YieldOp>(loc, loop_buf);

  _.restoreInsertionPoint(ip);
  return result_buf;
}

mlir::Value GeneratePass::transcodeInlineVector(mlir::Location loc,
                                                mlir::OpBuilder& _, Value src,
                                                Value dst, Value copy_length,
                                                Value buf) {
  auto src_type = src.getType().cast<VectorType>();
  auto dst_type = dst.getType().cast<VectorType>();

  _.create<AssumeOp>(
      loc, _.create<CmpIOp>(loc, CmpIPredicate::ule, copy_length,
                            buildIndex(loc, _, src_type->min_length)));

  Value is_dst_inline = [&]() -> Value {
    if (dst_type->min_length == 0) {
      return buildBool(loc, _, false);
    }
    return _.create<CmpIOp>(loc, CmpIPredicate::ule, copy_length,
                            buildIndex(loc, _, dst_type->min_length));
  }();

  auto zero = buildIndex(loc, _, 0);
  auto dst_inline_if = _.create<scf::IfOp>(loc, buf.getType(), is_dst_inline,
                                           /*withElseRegion=*/true);
  auto ip = _.saveInsertionPoint();

  // If dst is inline
  {
    _.setInsertionPointToStart(dst_inline_if.thenBlock());

    // Mark the dst inline region as poison.
    auto region_start = _.create<VectorIndexOp>(loc, dst_type->elem, dst, zero,
                                                VectorRegion::Inline, buf);
    auto region_size = dst_type->elem.headSize() * dst_type->min_length;
    _.create<PoisonOp>(loc, region_start, Bytes(0), region_size);

    _.create<scf::YieldOp>(
        loc, generateVectorCopyLoop(loc, _, src, zero, VectorRegion::Inline,
                                    dst, zero, VectorRegion::Inline,
                                    copy_length, buf, buf));
  }

  // If dst is outline
  {
    _.setInsertionPointToStart(dst_inline_if.elseBlock());

    Value aligned_buf = _.create<AlignOp>(
        loc, buf.getType(), buf, dst_type->elem.headAlignment().bytes());
    _.create<StoreRefOp>(loc, dst, aligned_buf);

    auto ppl_count = buildIndex(loc, _, dst_type->ppl_count);
    auto outline_count = _.create<SubIOp>(loc, copy_length, ppl_count);
    auto outline_bytes = _.create<MulIOp>(
        loc, buildIndex(loc, _, dst_type->elem_width.bytes()), outline_count);

    Value result_buf = _.create<AllocateOp>(loc, aligned_buf.getType(),
                                            aligned_buf, outline_bytes);
    result_buf = generateVectorCopyLoop(loc, _, src, zero, VectorRegion::Inline,
                                        dst, zero, VectorRegion::Partial,
                                        ppl_count, aligned_buf, result_buf);
    _.create<scf::YieldOp>(
        loc, generateVectorCopyLoop(
                 loc, _, src, ppl_count, VectorRegion::Inline, dst, zero,
                 VectorRegion::Buffer, outline_count, aligned_buf, result_buf));
  }

  _.restoreInsertionPoint(ip);
  return dst_inline_if.getResult(0);
}

mlir::Value GeneratePass::transcodeOutlineVector(mlir::Location loc,
                                                 mlir::OpBuilder& _, Value src,
                                                 Value dst, Value copy_length,
                                                 Value buf) {
  auto src_type = src.getType().cast<VectorType>();
  auto dst_type = dst.getType().cast<VectorType>();

  if (src_type->min_length > 0) {
    _.create<AssumeOp>(
        loc,
        _.create<CmpIOp>(
            loc, CmpIPredicate::uge, copy_length,
            buildIndex(
                loc, _,
                std::min(src_type->min_length + 1,
                         static_cast<uint64_t>(dst_type->maxLengthBound())))));
  }

  Value is_dst_inline = [&]() -> Value {
    if (dst_type->min_length == 0) {
      return buildBool(loc, _, false);
    }
    return _.create<CmpIOp>(loc, CmpIPredicate::ule, copy_length,
                            buildIndex(loc, _, dst_type->min_length));
  }();

  auto zero = buildIndex(loc, _, 0);
  auto dst_inline_if = _.create<scf::IfOp>(loc, buf.getType(), is_dst_inline,
                                           /*withElseRegion=*/true);
  auto ip = _.saveInsertionPoint();

  {  // dst is inline
    _.setInsertionPointToStart(dst_inline_if.thenBlock());

    // Mark the dst inline region as poison.
    auto region_start = _.create<VectorIndexOp>(loc, dst_type->elem, dst, zero,
                                                VectorRegion::Inline, buf);
    auto region_size = dst_type->elem.headSize() * dst_type->min_length;
    _.create<PoisonOp>(loc, region_start, Bytes(0), region_size);

    auto ppl_count = buildIndex(
        loc, _, std::min(src_type->ppl_count, dst_type->maxLengthBound()));
    auto outline_count = _.create<SubIOp>(loc, copy_length, ppl_count);

    auto result_buf =
        generateVectorCopyLoop(loc, _, src, zero, VectorRegion::Partial, dst,
                               zero, VectorRegion::Inline, ppl_count, buf, buf);
    _.create<scf::YieldOp>(
        loc, generateVectorCopyLoop(loc, _, src, zero, VectorRegion::Reference,
                                    dst, ppl_count, VectorRegion::Inline,
                                    outline_count, buf, result_buf));
  }

  {  // dst is outline
    _.setInsertionPointToStart(dst_inline_if.elseBlock());

    Value aligned_buf = _.create<AlignOp>(
        loc, buf.getType(), buf, dst_type->elem.headAlignment().bytes());
    _.create<StoreRefOp>(loc, dst, aligned_buf);

    auto src_ppl_count = buildIndex(loc, _, src_type->ppl_count);
    auto dst_ppl_count = buildIndex(loc, _, dst_type->ppl_count);

    if (src_type->ppl_count <= dst_type->ppl_count) {
      auto dst_ppl_leftover =
          buildIndex(loc, _, dst_type->ppl_count - src_type->ppl_count);
      auto outline_count = _.create<SubIOp>(loc, copy_length, dst_ppl_count);
      auto outline_bytes = _.create<MulIOp>(
          loc, buildIndex(loc, _, dst_type->elem_width.bytes()), outline_count);

      Value result_buf = _.create<AllocateOp>(loc, aligned_buf.getType(),
                                              aligned_buf, outline_bytes);
      result_buf = generateVectorCopyLoop(
          loc, _, src, zero, VectorRegion::Partial, dst, zero,
          VectorRegion::Partial, src_ppl_count, aligned_buf, result_buf);
      result_buf = generateVectorCopyLoop(
          loc, _, src, zero, VectorRegion::Reference, dst, src_ppl_count,
          VectorRegion::Partial, dst_ppl_leftover, aligned_buf, result_buf);
      _.create<scf::YieldOp>(
          loc, generateVectorCopyLoop(loc, _, src, dst_ppl_leftover,
                                      VectorRegion::Reference, dst, zero,
                                      VectorRegion::Buffer, outline_count,
                                      aligned_buf, result_buf));
    } else {
      auto src_ppl_leftover =
          buildIndex(loc, _, src_type->ppl_count - dst_type->ppl_count);
      auto dst_outline_count =
          _.create<SubIOp>(loc, copy_length, dst_ppl_count);
      auto src_outline_count =
          _.create<SubIOp>(loc, copy_length, src_ppl_count);
      auto dst_outline_bytes = _.create<MulIOp>(
          loc, buildIndex(loc, _, dst_type->elem_width.bytes()),
          dst_outline_count);

      Value result_buf = _.create<AllocateOp>(loc, aligned_buf.getType(),
                                              aligned_buf, dst_outline_bytes);
      result_buf = generateVectorCopyLoop(
          loc, _, src, zero, VectorRegion::Partial, dst, zero,
          VectorRegion::Partial, dst_ppl_count, aligned_buf, result_buf);
      result_buf = generateVectorCopyLoop(
          loc, _, src, dst_ppl_count, VectorRegion::Partial, dst, zero,
          VectorRegion::Buffer, src_ppl_leftover, aligned_buf, result_buf);
      _.create<scf::YieldOp>(
          loc,
          generateVectorCopyLoop(loc, _, src, zero, VectorRegion::Reference,
                                 dst, src_ppl_leftover, VectorRegion::Buffer,
                                 src_outline_count, aligned_buf, result_buf));
    }
  }

  _.restoreInsertionPoint(ip);
  return dst_inline_if.getResult(0);
}

mlir::FuncOp GeneratePass::getOrCreateReflectFn(mlir::Location loc,
                                                OpBuilder::Listener* listener,
                                                ValueType from, AnyType to,
                                                mlir::Type buf_type) {
  auto* ctx = &getContext();
  auto key =
      FnKey{from, to, PathAttr::none(ctx), ArrayAttr::get(ctx, {}), buf_type};
  auto func = getOrCreateFn(loc, listener, "xcd", key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
  _.setListener(listener);

  Value src = func.getArgument(0), dst = func.getArgument(1),
        buf = func.getArgument(2);

  auto reflected_type = reflect::reflectableTypeFor(
      from, ReflectDomainAttr::unique(_.getContext()));

  Value aligned_buf = _.create<AlignOp>(loc, buf.getType(), buf,
                                        reflected_type.headAlignment().bytes());

  auto reflected_dst =
      _.create<ProjectOp>(loc, reflected_type, aligned_buf, Bytes(0));

  Value result_buf = _.create<AllocateOp>(
      loc, aligned_buf.getType(), aligned_buf,
      buildIndex(loc, _, reflected_type.headSize().bytes()));

  result_buf = _.create<TranscodeOp>(
      loc, result_buf.getType(), src, reflected_dst, result_buf,
      PathAttr::get(ctx), ArrayAttr::get(ctx, {}));

  _.create<ReflectOp>(loc, reflected_dst, dst);
  _.create<ReturnOp>(loc, result_buf);

  return func;
}

mlir::FuncOp GeneratePass::getOrCreateVectorTranscodeFn(
    mlir::Location loc, OpBuilder::Listener* listener, VectorType from,
    VectorType to, mlir::Type buf_type) {
  auto* ctx = &getContext();
  auto key =
      FnKey{from, to, PathAttr::none(ctx), ArrayAttr::get(ctx, {}), buf_type};
  auto func = getOrCreateFn(loc, listener, "xcd", key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto* entry_block = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entry_block);
  _.setListener(listener);

  Value src = func.getArgument(0), dst = func.getArgument(1),
        buf = func.getArgument(2);

  auto src_length = _.create<LengthOp>(loc, _.getIndexType(), src);
  auto copy_length = [&]() -> Value {
    if (to->max_length < 0) {
      return src_length;
    }
    auto max = buildIndex(loc, _, to->max_length);
    auto cond = _.create<CmpIOp>(loc, CmpIPredicate::ule, src_length, max);
    return _.create<SelectOp>(loc, cond, src_length, max);
  }();

  _.create<StoreLengthOp>(loc, dst, copy_length);

  auto is_src_inline = [&]() -> Value {
    if (from->min_length == 0) {
      return buildBool(loc, _, false);
    }
    return _.create<CmpIOp>(loc, CmpIPredicate::ule, src_length,
                            buildIndex(loc, _, from->min_length));
  }();

  auto src_inline_if = _.create<scf::IfOp>(loc, buf_type, is_src_inline,
                                           /*withElseRegion=*/true);
  _.create<ReturnOp>(loc, src_inline_if.getResult(0));

  {  // src is inline
    _.setInsertionPointToStart(src_inline_if.thenBlock());
    _.create<scf::YieldOp>(
        loc, transcodeInlineVector(loc, _, src, dst, copy_length, buf));
  }

  {  // src is not inline
    _.setInsertionPointToStart(src_inline_if.elseBlock());
    _.create<scf::YieldOp>(
        loc, transcodeOutlineVector(loc, _, src, dst, copy_length, buf));
  }

  cached_fns_.emplace(key, func);
  return mlir::FuncOp{func};
}

struct EncodeFunctionLowering : public OpConversionPattern<EncodeFunctionOp> {
  using OpConversionPattern<EncodeFunctionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(EncodeFunctionOp op,
                                llvm::ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult EncodeFunctionLowering::matchAndRewrite(
    EncodeFunctionOp op, llvm::ArrayRef<Value> operands,
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

  auto src =
      _.create<ProjectOp>(loc, op.src(), func.getArgument(0), Bits(0), true);

  auto dst =
      _.create<ProjectOp>(loc, proto->head, func.getArgument(1), Bits(0));

  auto dst_buf =
      _.create<ProjectOp>(loc, RawBufferType::get(ctx), func.getArgument(1),
                          proto->head.headSize() + proto->buffer_offset);

  _.create<TranscodeOp>(loc, RawBufferType::get(ctx), src, dst, dst_buf,
                        op.src_path(), ArrayAttr::get(ctx, {}));

  _.create<ReturnOp>(loc);

  return success();
}

struct DecodeFunctionLowering : public OpConversionPattern<DecodeFunctionOp> {
  using OpConversionPattern<DecodeFunctionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(DecodeFunctionOp op,
                                llvm::ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult DecodeFunctionLowering::matchAndRewrite(
    DecodeFunctionOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  _.eraseOp(op);

  auto loc = op.getLoc();
  auto* ctx = _.getContext();

  const auto& proto = op.src().cast<ProtocolType>();

  // Create a function with the given name, accepting a protocol, destination
  // type, buffer, handler array, and user-state pointer. The function contains
  // a single DecodeOp.
  auto func = mlir::FuncOp::create(
      loc, op.name(),
      _.getFunctionType(
          {op.src(), op.dst(), types::BoundedBufferType::get(getContext()),
           HandlersArrayType::get(getContext()),
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

  auto src = _.create<ir::ProjectOp>(loc, proto->head, func.getArgument(0),
                                     Bits(0), true);

  Value buf = _.create<TranscodeOp>(loc, BoundedBufferType::get(ctx), src,
                                    func.getArgument(1), func.getArgument(2),
                                    PathAttr::none(ctx), op.handlers());

  _.create<InvokeCallbackOp>(loc, func.getArgument(1), func.getArgument(3),
                             func.getArgument(4));

  _.create<YieldOp>(loc, buf);

  return success();
}

struct SizeFunctionLowering : public OpConversionPattern<SizeFunctionOp> {
  using OpConversionPattern<SizeFunctionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SizeFunctionOp op,
                                llvm::ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult SizeFunctionLowering::matchAndRewrite(
    SizeFunctionOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  _.eraseOp(op);

  auto loc = op.getLoc();
  auto* ctx = _.getContext();

  const auto& proto = op.dst().cast<ProtocolType>();

  // Create a function with the given name taking a pointer to the source type.
  // The function contains a single SizeOp.
  auto func = mlir::FuncOp::create(
      loc, op.name(), _.getFunctionType({op.src()}, _.getIndexType()));

  mlir::ModuleOp module{op.getOperation()->getParentOp()};
  module.push_back(func);
  auto* entry_block = func.addEntryBlock();
  _.setInsertionPointToStart(entry_block);

  // Create a SizeOp to compute the size from TranscodeOps
  auto size = _.create<SizeOp>(loc, _.getIndexType(), op.round_up());
  auto head_size = _.create<ConstantOp>(
      loc, _.getIntegerAttr(_.getIndexType(), proto->headSize().bytes()));
  _.create<ReturnOp>(loc, ValueRange{_.create<AddIOp>(loc, size, head_size)});

  auto* body = new Block();
  body->addArgument(DummyBufferType::get(ctx));
  size.body().push_back(body);

  _.setInsertionPointToStart(body);

  Value buf = body->getArgument(0);
  auto dst = _.create<ir::UnitOp>(loc, proto->head);

  buf =
      _.create<TranscodeOp>(loc, DummyBufferType::get(ctx), func.getArgument(0),
                            dst, buf, op.src_path(), ArrayAttr::get(ctx, {}));
  _.create<YieldOp>(loc, buf);

  return success();
}

struct TranscodeOpLowering : public OpConversionPattern<TranscodeOp> {
  TranscodeOpLowering(MLIRContext* ctx, GeneratePass* pass)
      : OpConversionPattern<TranscodeOp>(ctx), pass(pass) {
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(TranscodeOp op, llvm::ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  GeneratePass* pass;
};

LogicalResult TranscodeOpLowering::matchAndRewrite(
    TranscodeOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto loc = op.getLoc();
  auto src_type = op.src().getType().cast<ValueType>();
  auto dst_type = op.dst().getType().cast<ValueType>();

  if (dst_type.isa<UnitType>()) {
    _.replaceOp(op, op.buf());
    return success();
  }

  if (src_type.isa<UnitType>()) {
    _.create<DefaultOp>(loc, op.buf().getType(), op.dst(), op.buf(),
                        op.handlers());
    _.replaceOp(op, op.buf());
    return success();
  }

  if (src_type.isa<PrimitiveType>() && dst_type.isa<PrimitiveType>() &&
      src_type.getTypeID() == dst_type.getTypeID()) {
    _.create<TranscodePrimitiveOp>(loc, operands[0], operands[1]);
    _.replaceOp(op, operands[2]);
    return success();
  }

  FuncOp fn;

  if (src_type.isa<StructType>() && dst_type.isa<StructType>()) {
    fn = pass->getOrCreateStructTranscodeFn(
        loc, _.getListener(), src_type.cast<StructType>(),
        dst_type.cast<StructType>(), op.path(), op.handlers(),
        op.getResult().getType());
  } else if (src_type.isa<VariantType>() && dst_type.isa<VariantType>()) {
    auto src_var = src_type.cast<VariantType>(),
         dst_var = dst_type.cast<VariantType>();
    if (src_type.isEnum() && dst_type.isEnum()) {
      bool exact_match = src_var.terms().size() == dst_var.terms().size();
      uint64_t max_src_tag = 0;
      for (size_t i = 0; i < src_var.terms().size(); ++i) {
        max_src_tag = std::max(max_src_tag, src_var.terms()[i].tag);
        if (exact_match &&
            (src_var.terms()[i].name != dst_var.terms()[i].name ||
             src_var.terms()[i].tag != dst_var.terms()[i].tag)) {
          exact_match = false;
        }
      }

      // Check that tag values aren't so large that we build an
      // absurd lookup table.
      static constexpr size_t kEnumTableExpansionLimit = 4;
      if (exact_match ||
          max_src_tag <= kEnumTableExpansionLimit *
                             src_type.cast<VariantType>().terms().size()) {
        _.create<CopyTagOp>(loc, operands[0], operands[1]);
        _.replaceOp(op, operands[2]);
        return success();
      }
    }

    fn = pass->getOrCreateVariantTranscodeFn(
        loc, _.getListener(), src_type.cast<VariantType>(),
        dst_type.cast<VariantType>(), op.path(), op.handlers(),
        op.getResult().getType());
  } else if (src_type.isa<ArrayType>() && dst_type.isa<ArrayType>()) {
    fn = pass->getOrCreateArrayTranscodeFn(
        loc, _.getListener(), src_type.cast<ArrayType>(),
        dst_type.cast<ArrayType>(), op.getResult().getType());
  } else if (src_type.isa<VectorType>() && dst_type.isa<VectorType>()) {
    fn = pass->getOrCreateVectorTranscodeFn(
        loc, _.getListener(), src_type.cast<VectorType>(),
        dst_type.cast<VectorType>(), op.getResult().getType());
  } else if (auto any = dst_type.dyn_cast<AnyType>()) {
    fn = pass->getOrCreateReflectFn(loc, _.getListener(), src_type, any,
                                    op.getResult().getType());
  }

  auto call = _.create<mlir::CallOp>(loc, fn, operands);
  _.replaceOp(op, call.getResult(0));
  return success();
}

struct DefaultOpLowering : public OpConversionPattern<DefaultOp> {
  DefaultOpLowering(MLIRContext* ctx) : OpConversionPattern<DefaultOp>(ctx) {
    setHasBoundedRewriteRecursion(true);
  }

  LogicalResult matchAndRewrite(DefaultOp op, llvm::ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;
};

LogicalResult DefaultOpLowering::matchAndRewrite(
    DefaultOp op, llvm::ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto* ctx = _.getContext();
  auto loc = op.getLoc();
  auto type = op.dst().getType();

  if (type.isa<UnitType>()) {
    _.eraseOp(op);
    return success();
  }

  UnitOp src;

  if (type.isa<StructType>()) {
    llvm::StringRef name = "<empty>";
    src = _.create<UnitOp>(
        loc, StructType::get(ctx, HostDomainAttr::get(ctx), &name));
  } else if (type.isa<VariantType>()) {
    llvm::StringRef name = "<empty>";
    src = _.create<UnitOp>(
        loc, InlineVariantType::get(ctx, HostDomainAttr::get(ctx), &name));
  } else if (type.isa<ArrayType>()) {
    src = _.create<UnitOp>(
        loc, ArrayType::get(ctx, Array{.elem = type.cast<ArrayType>()->elem}));
  } else {
    return failure();
  }

  _.create<TranscodeOp>(loc, op.buf().getType(), src, op.dst(), op.buf(),
                        PathAttr::none(ctx), op.handlers());
  _.eraseOp(op);
  return success();
}

void GeneratePass::runOnOperation() {
  auto* ctx = &getContext();

  ConversionTarget target(*ctx);
  target
      .addLegalDialect<StandardOpsDialect, scf::SCFDialect, ProtoJitDialect>();
  target.addLegalOp<ModuleOp, FuncOp>();
  target.addIllegalOp<EncodeFunctionOp, DecodeFunctionOp, SizeFunctionOp,
                      TranscodeOp>();
  target.addDynamicallyLegalOp<DefaultOp>([](DefaultOp op) {
    return op.dst().getType().isa<PrimitiveType>() ||
           op.dst().getType().isa<VectorType>();
  });

  OwningRewritePatternList patterns(ctx);
  patterns
      .insert<EncodeFunctionLowering, DecodeFunctionLowering,
              SizeFunctionLowering, DefaultOpLowering>(ctx)
      .insert<TranscodeOpLowering>(ctx, this);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createIRGenPass() {
  return std::make_unique<GeneratePass>();
}

}  // namespace pj
