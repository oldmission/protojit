#include <unordered_map>
#include <unordered_set>

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include "defer.hpp"
#include "ir.hpp"
#include "util.hpp"

namespace pj {
struct EncodeFnKey {
  types::ValueType from;
  types::ValueType to;
  types::PathAttr path;

  bool operator==(const EncodeFnKey& other) const {
    return from == other.from && to == other.to && path == other.path;
  }
};
}  // namespace pj

namespace std {
template <>
struct hash<pj::EncodeFnKey> {
  size_t operator()(const pj::EncodeFnKey& key) const {
    using ::llvm::hash_value;
    return llvm::hash_combine(hash_value(key.from), hash_value(key.to),
                              hash_value(key.path));
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
  mlir::FuncOp getOrCreateStructEncodeFn(mlir::Location loc, StructType from,
                                         StructType to, PathAttr path);

  mlir::FuncOp getOrCreateVariantEncodeFn(mlir::Location loc, VariantType from,
                                          VariantType to, PathAttr path);

  mlir::ModuleOp module() { return mlir::ModuleOp(getOperation()); }

 private:
  void encodeTerm(mlir::OpBuilder& _, mlir::Location loc, VariantType src_type,
                  VariantType dst_type, const Term* src_term,
                  const Term* dst_term, Value src, Value dst, Value& buffer);

  mlir::FuncOp getOrCreateEncodeFn(mlir::Location loc, const EncodeFnKey& key);

  std::unordered_map<EncodeFnKey, Operation*> encode_fns;
  std::unordered_map<std::string, std::unordered_set<uint32_t>> used_names;
};

void GeneratePass::encodeTerm(mlir::OpBuilder& _, mlir::Location loc,
                              VariantType src_type, VariantType dst_type,
                              const Term* src_term, const Term* dst_term,
                              Value src, Value dst, Value& buffer) {
  if (dst_term != nullptr) {
    auto src_body = _.create<ir2::ProjectOp>(loc, src_term->type, src,
                                             src_type.term_offset());
    auto dst_body = _.create<ir2::ProjectOp>(loc, dst_term->type, dst,
                                             dst_type.term_offset());

    if (dst_type.isa<OutlineVariantType>()) {
      buffer = _.create<ir2::ProjectOp>(
          loc, types::RawBufferType::get(&getContext()), buffer,
          dst_term->type.cast<ValueType>().head_size());
    }

    _.create<ir2::EncodeOp>(loc, types::RawBufferType::get(&getContext()),
                            src_body, dst_body, buffer,
                            types::PathAttr::none(&getContext()));
  }

  _.create<TagOp>(loc, dst,
                  dst_term == nullptr ? VariantType::kUndefTag : dst_term->tag);
}

mlir::FuncOp GeneratePass::getOrCreateEncodeFn(mlir::Location loc,
                                               const EncodeFnKey& key) {
  if (auto it = encode_fns.find(key); it != encode_fns.end()) {
    return mlir::FuncOp{it->second};
  }

  mlir::OpBuilder _ = mlir::OpBuilder::atBlockBegin(module().getBody());

  std::string name;
  {
    // TODO: use twine for all these concats.
    llvm::raw_string_ostream os(name);

    // Start all internal functions with a '#', so we can identify
    // them for internal linkage later.
    os << "#enc_";
    key.from.print(os);
    os << "_to_";
    key.to.print(os);
    os << "_at_";
    key.path.print(os);

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
  return _.create<mlir::FuncOp>(
      loc, name,
      _.getFunctionType(
          {key.from, key.to, types::RawBufferType::get(&getContext())},
          types::RawBufferType::get(&getContext())));
}

mlir::FuncOp GeneratePass::getOrCreateStructEncodeFn(mlir::Location loc,
                                                     StructType from,
                                                     StructType to,
                                                     PathAttr path) {
  auto key = EncodeFnKey{from, to, path};
  auto func = getOrCreateEncodeFn(loc, key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto entryBlock = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entryBlock);

  auto src = func.getArgument(0);
  auto dst = func.getArgument(1);

  // Encode each field in the target struct.
  // TODO: cache this map from names to fields in the source struct?
  std::map<std::string, const StructField*> from_fields;
  for (auto& field : from->fields) {
    from_fields.emplace(field.name.str(), &field);
  }

  Value result_buf = func.getArgument(2);
  for (auto& to_field : to->fields) {
    // Project out the target field.
    auto dst_field =
        _.create<ir2::ProjectOp>(loc, to_field.type, dst, to_field.offset);
    if (auto it = from_fields.find(std::string{to_field.name});
        it != from_fields.end()) {
      auto& from_field = it->second;
      auto src_field = _.create<ir2::ProjectOp>(loc, from_field->type, src,
                                                from_field->offset);

      // If the target field exists in the source struct, encode
      // the source field into the target.
      result_buf = _.create<ir2::EncodeOp>(
          loc, RawBufferType::get(&getContext()), src_field, dst_field,
          result_buf, path.into(to_field.name));
    } else {
      // Otherwise fill in the target field with a default value.
      _.create<ir2::DefaultOp>(loc, dst_field);
    }
  }

  _.create<ReturnOp>(loc, result_buf);

  encode_fns.emplace(key, func);
  return mlir::FuncOp{func};
}

mlir::FuncOp GeneratePass::getOrCreateVariantEncodeFn(mlir::Location loc,
                                                      VariantType src_type,
                                                      VariantType dst_type,
                                                      PathAttr path) {
  auto key = EncodeFnKey{src_type, dst_type, path};
  auto func = getOrCreateEncodeFn(loc, key);

  if (!func.isDeclaration()) {
    return func;
  }

  auto entryBlock = func.addEntryBlock();
  auto _ = mlir::OpBuilder::atBlockBegin(entryBlock);

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
      encodeTerm(_, loc, src_type, dst_type, src_term, it->second, src, dst,
                 result_buf);
    } else {
      encodeTerm(_, loc, src_type, dst_type, src_term, nullptr, src, dst,
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

      encodeTerm(_, loc, src_type, dst_type, nullptr, nullptr, src, dst,
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
        encodeTerm(_, loc, src_type, dst_type, &src_term, it->second, src, dst,
                   result_buf);
      } else {
        encodeTerm(_, loc, src_type, dst_type, &src_term, nullptr, src, dst,
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

  encode_fns.emplace(key, func);
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

  // Mark both arguments to the function as "noalias" to aid
  // LLVM optimization.
  func.setArgAttr(0, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(C));
  func.setArgAttr(1, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(C));

  mlir::ModuleOp module{op.getOperation()->getParentOp()};
  module.push_back(func);
  auto entryBlock = func.addEntryBlock();
  _.setInsertionPointToStart(entryBlock);

  auto dst =
      _.create<ir2::ProjectOp>(L, proto->head, func.getArgument(1), Bits(0));

  auto dst_buf = _.create<ir2::ProjectOp>(
      L, RawBufferType::get(C), func.getArgument(1), proto->head.head_size());

  _.create<EncodeOp>(L, RawBufferType::get(C), func.getArgument(0), dst,
                     dst_buf, op.src_path());

  _.create<ReturnOp>(L);

  return success();
}

struct EncodeOpLowering : public OpConversionPattern<EncodeOp> {
  EncodeOpLowering(MLIRContext* C, GeneratePass* pass)
      : OpConversionPattern<EncodeOp>(C), pass(pass) {}

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
        L, src_type.cast<StructType>(), dst_type.cast<StructType>(), op.path());
    auto call = _.create<mlir::CallOp>(L, fn, operands);
    _.replaceOp(op, call.getResult(0));
    return success();
  }

  // Call an outlined encoding routine for variants.
  if (src_type.isa<VariantType>() && dst_type.isa<VariantType>()) {
    auto fn = pass->getOrCreateVariantEncodeFn(L, src_type.cast<VariantType>(),
                                               dst_type.cast<VariantType>(),
                                               op.path());
    auto call = _.create<mlir::CallOp>(L, fn, operands);
    _.replaceOp(op, call.getResult(0));
    return success();
  }

  return failure();
}

void GeneratePass::runOnOperation() {
  auto* C = &getContext();

  ConversionTarget target(*C);
  target.addLegalDialect<StandardOpsDialect, ProtoJitDialect>();
  target.addIllegalOp<EncodeFunctionOp, DecodeFunctionOp, SizeFunctionOp,
                      EncodeOp>();

  OwningRewritePatternList patterns(C);
  patterns.insert<EncodeFunctionLowering>(C).insert<EncodeOpLowering>(C, this);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createGeneratePass() {
  return std::make_unique<GeneratePass>();
}

}  // namespace pj
