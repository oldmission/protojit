#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "concrete_types.hpp"
#include "defer.hpp"
#include "ir.hpp"
#include "tag.hpp"

namespace pj {
using namespace ir;

mlir::Value CIntType::GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,
                                PathPiece path, const CType* cto,
                                mlir::OpBuilder& _, const mlir::Value& from,
                                const mlir::Value& to) const {
  assert(cto->IsInt());
  _.create<XIntOp>(from.getLoc(), from, to);

  // Doesn't make a tag.
  return nullptr;
}

mlir::Value CStructType::GenEncode(Scope* S, mlir::MLIRContext* C,
                                   PathPiece tag, PathPiece path,
                                   const CType* cto_, mlir::OpBuilder& _,
                                   const mlir::Value& from,
                                   const mlir::Value& to) const {
  const auto* cto = cto_->AsStruct();
  auto L = from.getLoc();

  auto xstr = _.create<XStrOp>(L, from, to);
  DEFER(_.setInsertionPointAfter(xstr));

  auto entry = new Block();
  xstr.body().push_back(entry);
  _.setInsertionPointToStart(entry);

  Value tagv{};

  // Encode shared fields.
  // TODO(6): handle aliases
  for (const auto& [name, from_field] : fields) {
    if (auto it = cto->fields.find(name); it != cto->fields.end()) {
      const auto& [name, to_field] = *it;

      auto from_inner =
          _.create<ProjectOp>(from.getLoc(), from_field.type->toIR(C), from,
                              from_field.offset.bytes());
      auto to_inner = _.create<ProjectOp>(from.getLoc(), to_field.type->toIR(C),
                                          to, to_field.offset.bytes());

      Value inner_tag = from_field.type->GenEncode(
          S, C, Narrow(tag, name), Narrow(path, name), to_field.type, _,
          from_inner, to_inner);

      if (inner_tag != nullptr) {
        assert(!tagv);
        tagv = inner_tag;
      }
    }
  }

  // Fill in unknown fields on the other side with defaults.
  for (auto& [name, to_field] : cto->fields) {
    if (fields.count(name)) continue;

    auto to_inner = _.create<ProjectOp>(from.getLoc(), to_field.type->toIR(C),
                                        to, to_field.offset.bytes());
    auto inner_tag = to_field.type->GenDefault(C, from.getLoc(), _, to_inner);

    if (inner_tag) {
      assert(!tagv);
      tagv = inner_tag;
    }
  }

  _.create<RetOp>(L);
  return tagv;
}

void CVariantType::GenEncodeForTerm(pj::Scope* S, mlir::MLIRContext* C,
                                    PathPiece tag, const std::string& head,
                                    PathPiece path, const CVariantType* cto,
                                    mlir::OpBuilder& _, const mlir::Value& from,
                                    const mlir::Value& to) const {
  auto L = from.getLoc();
  // TODO(6): aliases

  // If there is no matching term in the target ctype, encode
  // the "undef" term, whose tag is always 0.
  intptr_t encode_tag = kUndefTag;

  if (cto->terms.count(head)) {
    auto& from_term = terms.at(head);
    auto& to_term = cto->terms.at(head);
    encode_tag = to_term.tag;

    // Project inner values
    Value from_inner = _.create<ProjectOp>(L, from_term.type->toIR(C), from,
                                           term_offset.bytes());
    Value to_inner = _.create<ProjectOp>(L, to_term.type->toIR(C), to,
                                         cto->term_offset.bytes());

    from_term.type->GenEncode(S, C, {}, Narrow(path, head), to_term.type, _,
                              from_inner, to_inner);
  }

  const auto tag_val = GetIntegerConstant(L, _, cto->tag_size, encode_tag);

  if (cto->tag_offset.IsNotNone()) {
    _.create<ETagOp>(L, to, tag_val, cto->tag_offset);
  }

  if (IsDotTag(tag)) {
    _.create<RetOp>(L, tag_val);
  } else {
    _.create<RetOp>(L);
  }
}

mlir::Value CVariantType::GenEncode(pj::Scope* S, mlir::MLIRContext* C,
                                    PathPiece tag, PathPiece path,
                                    const CType* cto_, mlir::OpBuilder& _,
                                    const mlir::Value& from,
                                    const mlir::Value& to) const {
  auto cto = cto_->AsVariant();

  assert(IsEmptyTag(tag) || IsDotTag(tag));
  assert(cto->tag_size.IsNotNone());

  const auto L = from.getLoc();
  const auto joint_tag_size = IsDotTag(tag) ? cto->tag_size : Bits(0);

  if (MatchesExactlyAsEnum(cto)) {
    assert(joint_tag_size.bits() == 0);
    auto op = _.create<MatchVariantOp>(L, TypeRange{}, from);
    DEFER(_.setInsertionPointAfter(op));

    auto body = &op.body();
    auto entry = new Block();
    body->push_back(entry);

    _.setInsertionPointToStart(entry);
    auto from_tag = _.create<ProjectOp>(L, GetTagAsIntegerType(S, C), from,
                                        tag_offset.bytes());
    auto to_tag = _.create<ProjectOp>(L, cto->GetTagAsIntegerType(S, C), to,
                                      tag_offset.bytes());
    _.create<XIntOp>(L, from_tag, to_tag);
    _.create<RetOp>(L);

    return op.getResult(0);
  }

  MatchVariantOp op{};
  Region* body = nullptr;
  if (IsDotTag(tag)) {
    op = _.create<MatchVariantOp>(
        L, TypeRange{_.getIntegerType(joint_tag_size.bits())}, from);
  } else {
    op = _.create<MatchVariantOp>(L, TypeRange{}, from);
  }

  body = &op.body();

  DEFER(_.setInsertionPointAfter(op));

  auto entry = new Block();
  body->push_back(entry);
  _.setInsertionPointToStart(entry);

  if (path.begin != path.end) {
    const std::string term{*path.begin};
    if (term == "undef" or terms.count(term)) {
      GenEncodeForTerm(S, C, tag, term, path, cto, _, from, to);
    } else {
      llvm::errs() << "Cannot find term " << term << "\n";
      assert(terms.count(term));
    }
  } else {
    // The term hasn't been specified, so match on the tag value.
    // For each term that is known to the protocol, generate a BB
    // which encodes it. All other terms branch to a BB which encodes
    // an undefined term.
    assert(tag_offset.IsNotNone());

    // Canonicalization will clean up the undef block if it's not used.
    auto* undef = new Block();
    body->push_back(undef);
    _.setInsertionPointToStart(undef);
    GenEncodeForTerm(S, C, tag, "", path, cto, _, from, to);

    llvm::SmallVector<Block*, 8> successors;
    // TODO(25): limits tag sizes
    llvm::SmallVector<uint64_t, 8> tags;

    // Add a case for the source being undefined as well.
    successors.push_back(undef);
    tags.push_back(kUndefTag);

    for (auto& term : terms) {
      if (cto->terms.count(term.first)) {
        auto* block = new Block();
        body->push_back(block);
        successors.push_back(block);
        _.setInsertionPointToStart(block);
        GenEncodeForTerm(S, C, tag, term.first, path, cto, _, from, to);
      } else {
        successors.push_back(undef);
      }
      tags.push_back(term.second.tag);
    }

    _.setInsertionPointToEnd(entry);
    auto match_tag = _.create<DTagOp>(L, _.getIntegerType(tag_size.bits()),
                                      from, tag_offset);
    _.create<BTagOp>(L, match_tag, llvm::makeArrayRef(tags), successors);
  }

  return op.getNumResults() > 0 ? op.getResult(0) : mlir::Value{};
}

mlir::Value CArrayType::GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,
                                  PathPiece path, const CType* cto_,
                                  mlir::OpBuilder& _, const mlir::Value& from,
                                  const mlir::Value& to) const {
  auto* cto = cto_->AsArray();

  auto L = from.getLoc();

  auto op = _.create<XArrayOp>(L, from, to);
  DEFER(_.setInsertionPointAfter(op));

  auto entry = new Block();
  op.xvalue().push_back(entry);

  auto from_arg = entry->addArgument(el()->toIR(C));
  auto to_arg = entry->addArgument(cto->el()->toIR(C));

  // TODO: validate?
  assert(IsEmptyTag(tag));

  _.setInsertionPointToEnd(entry);
  el()->GenEncode(S, C, tag, path, cto->el(), _, from_arg, to_arg);

  _.create<RetOp>(L);

  entry = new Block();
  op.xdefault().push_back(entry);

  _.setInsertionPointToEnd(entry);
  to_arg = entry->addArgument(cto->el()->toIR(C));
  cto->el()->GenDefault(C, L, _, to_arg);
  _.create<RetOp>(L);

  return {};
}

mlir::Value CAnyType::GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,
                                PathPiece path, const CType* cto,
                                mlir::OpBuilder& builder,
                                const mlir::Value& from,
                                const mlir::Value& to) const {
  throw IssueError(17);
}

mlir::Value CListType::GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,
                                 PathPiece path, const CType* cto,
                                 mlir::OpBuilder& builder,
                                 const mlir::Value& from,
                                 const mlir::Value& to) const {
  throw IssueError(20);
}

mlir::Value COutlinedType::GenEncode(Scope* S, mlir::MLIRContext* C,
                                     PathPiece tag, PathPiece path,
                                     const CType* cto, mlir::OpBuilder& builder,
                                     const mlir::Value& from,
                                     const mlir::Value& to) const {
  throw IssueError(21);
}

mlir::Value CNamedType::GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,
                                  PathPiece path, const CType* cto,
                                  mlir::OpBuilder& builder,
                                  const mlir::Value& from,
                                  const mlir::Value& to) const {
  assert(false && "CNamedType is only used in sourcegen.");
  return nullptr;
}

}  // namespace pj
