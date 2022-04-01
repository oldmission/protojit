#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include "concrete_types.hpp"
#include "defer.hpp"
#include "exceptions.hpp"
#include "ir.hpp"
#include "tag.hpp"
#include "target.hpp"

namespace pj {
using namespace ir;

void CIntType::GenDecode(Scope* S, mlir::MLIRContext* C,
                         const DecodeTarget& target, PathPiece tag,
                         PathPiece dispatch, const CIntType* cto,
                         mlir::OpBuilder& _, const mlir::Value& base,
                         const mlir::Value& tagv, const mlir::Value& from,
                         const mlir::Value& to,
                         const mlir::Value& state) const {
  // No dispatching can be done on an int, just transcode it.
  _.create<XIntOp>(from.getLoc(), from, to);
}

void CStructType::GenDecode(Scope* S, mlir::MLIRContext* C,
                            const DecodeTarget& target, PathPiece tag,
                            PathPiece dispatch, const CStructType* cto,
                            mlir::OpBuilder& _, const mlir::Value& base,
                            const mlir::Value& tagv, const mlir::Value& from,
                            const mlir::Value& to,
                            const mlir::Value& state) const {
  auto L = from.getLoc();

  auto xstr = _.create<XStrOp>(L, from, to);
  DEFER(_.setInsertionPointAfter(xstr));

  auto entry = new Block();
  xstr.body().push_back(entry);
  _.setInsertionPointToStart(entry);

  auto handle_field = [&](auto& name, auto& from_field, auto& to_field) {
    auto from_inner =
        _.create<ProjectOp>(from.getLoc(), from_field.type->toIR(C), from,
                            from_field.offset.bytes());
    auto to_inner = _.create<ProjectOp>(from.getLoc(), to_field.type->toIR(C),
                                        to, to_field.offset.bytes());

    from_field.type->GenDecode(S, C, target, Narrow(tag, name),
                               Narrow(dispatch, name), to_field.type, _, base,
                               tagv, from_inner, to_inner, state);
  };

  std::string dispatch_field = "";
  if (!IsEmptyTag(dispatch)) {
    dispatch_field = *dispatch.begin;
    assert(fields.count(dispatch_field));
  }

  // TODO(6): handle aliases
  for (const auto& [name, from_field] : fields) {
    if (name != dispatch_field) {
      if (auto it = cto->fields.find(name); it != cto->fields.end()) {
        handle_field(name, from_field, it->second);
      }
    }
  }

  if (auto it = fields.find(dispatch_field); it != fields.end()) {
    if (auto jt = cto->fields.find(dispatch_field); jt != cto->fields.end()) {
      handle_field(dispatch_field, it->second, jt->second);
    }
  }

  // Fill in unknown fields on the other side with defaults.
  for (auto& [name, to_field] : cto->fields) {
    if (fields.count(name)) continue;

    auto to_inner = _.create<ProjectOp>(from.getLoc(), to_field.type->toIR(C),
                                        to, to_field.offset.bytes());
    to_field.type->GenDefault(C, from.getLoc(), _, to_inner);
  }

  _.create<RetOp>(L);
}

void CVariantType::GenDecodeForTerm(
    pj::Scope* S, mlir::MLIRContext* C, const DecodeTarget& target,
    PathPiece tag, PathPiece dispatch, const std::string& head,
    const CVariantType* cto, mlir::OpBuilder& _, const mlir::Value& base,
    const mlir::Value& tagv, const mlir::Value& from, const mlir::Value& to,
    const mlir::Value& state) const {
  auto L = to.getLoc();

  intptr_t encode_tag = kUndefTag;
  if (cto->terms.count(head)) {
    const CTerm& from_term = terms.at(head);
    const CTerm& to_term = cto->terms.at(head);
    encode_tag = to_term.tag;

    // TODO(9): support nested tags. This entails extracting the nested type's
    // tag from the joint tag.
    assert(IsEmptyTag(tag) or IsDotTag(tag));
    assert(IsEmptyTag(dispatch) or IsDotTag(dispatch));
    Value inner_tag_val{};

    const Value from_inner = _.create<ProjectOp>(L, from_term.type->toIR(C),
                                                 from, term_offset.bytes());
    const Value to_inner = _.create<ProjectOp>(L, to_term.type->toIR(C), to,
                                               cto->term_offset.bytes());

    from_term.type->GenDecode(S, C, target, PathPiece{}, PathPiece{},
                              to_term.type, _, base, inner_tag_val, from_inner,
                              to_inner, state);
  }

  if (!cto->tag_offset.IsNone()) {
    auto tag_val = GetIntegerConstant(L, _, cto->tag_size, encode_tag);

    // TODO(3): architecture-dependent
    _.create<ETagOp>(L, to, tag_val, cto->tag_offset);
  }

  if (IsDotTag(dispatch)) {
    if (auto it = target.branches.find(head); it != target.branches.end()) {
      auto target = it->second;

      // TODO(3): architecture-dependent
      auto target_type = _.getIntegerType(64, /*is_signed=*/false);
      auto target_attr =
          _.getIntegerAttr(target_type, reinterpret_cast<uint64_t>(target));
      _.create<DispatchOp>(L, base, state, target_attr);
    }
  }

  _.create<RetOp>(L);
}

void CVariantType::GenDecode(Scope* S, mlir::MLIRContext* C,
                             const DecodeTarget& target, PathPiece tag,
                             PathPiece dispatch, const CVariantType* cto,
                             mlir::OpBuilder& _, const mlir::Value& base,
                             const mlir::Value& tagv, const mlir::Value& from,
                             const mlir::Value& to,
                             const mlir::Value& state) const {
  // The tag has to come from somewhere.
  assert(IsDotTag(tag) || terms.size() == 0 || tag_offset.IsNotNone());

  auto L = from.getLoc();

  auto op = _.create<MatchVariantOp>(L, TypeRange{}, from);
  Region* body = &op.body();

  DEFER(_.setInsertionPointAfter(op));

  auto entry = new Block();
  body->push_back(entry);
  _.setInsertionPointToStart(entry);

  llvm::SmallVector<Block*, 8> successors;
  llvm::SmallVector<uint64_t, 8> tags;

  auto* undef = new Block();
  body->push_back(undef);
  _.setInsertionPointToStart(undef);
  GenDecodeForTerm(S, C, target, tag, dispatch, "undef", cto, _, base, tagv,
                   from, to, state);

  successors.push_back(undef);
  tags.push_back(kUndefTag);

  assert(not terms.count("undef"));
  for (auto& [tname, ttag] : terms) {
    if (cto->terms.count(tname)) {
      auto* block = new Block();
      body->push_back(block);
      successors.push_back(block);
      _.setInsertionPointToStart(block);
      GenDecodeForTerm(S, C, target, tag, dispatch, tname, cto, _, base, tagv,
                       from, to, state);
    } else {
      successors.push_back(undef);
    }
    tags.push_back(ttag.tag);
  }

  _.setInsertionPointToEnd(entry);

  // Load the tag from the joint tag, if available.
  Value match_tag{};
  if (IsDotTag(tag)) {
    // TODO(9): nested tags
    match_tag = tagv;
  } else {
    match_tag = _.create<DTagOp>(L, _.getIntegerType(tag_size.bits()), from,
                                 tag_offset);
  }
  _.create<BTagOp>(L, match_tag, llvm::makeArrayRef(tags), successors);
}

void CArrayType::GenDecode(Scope* S, mlir::MLIRContext* C,
                           const DecodeTarget& target, PathPiece dispatch,
                           PathPiece tag, const CArrayType* cto,
                           mlir::OpBuilder& _, const mlir::Value& base,
                           const mlir::Value& tagv, const mlir::Value& from,
                           const mlir::Value& to,
                           const mlir::Value& state) const {
  auto L = from.getLoc();

  auto op = _.create<XArrayOp>(L, from, to);
  DEFER(_.setInsertionPointAfter(op));

  auto entry = new Block();
  op.xvalue().push_back(entry);

  auto from_arg = entry->addArgument(el()->toIR(C));
  auto to_arg = entry->addArgument(cto->el()->toIR(C));

  _.setInsertionPointToEnd(entry);
  el()->GenDecode(S, C, target, dispatch, tag, cto->el(), _, base, tagv,
                  from_arg, to_arg, state);

  _.create<RetOp>(L);

  entry = new Block();
  op.xdefault().push_back(entry);

  _.setInsertionPointToEnd(entry);
  to_arg = entry->addArgument(cto->el()->toIR(C));
  cto->el()->GenDefault(C, L, _, to_arg);
  _.create<RetOp>(L);
}

void CListType::GenDecode(Scope* S, mlir::MLIRContext* C,
                          const DecodeTarget& target, PathPiece dispatch,
                          PathPiece tag, const CListType* cto,
                          mlir::OpBuilder& builder, const mlir::Value& base,
                          const mlir::Value& tagv, const mlir::Value& from,
                          const mlir::Value& to,
                          const mlir::Value& state) const {
  throw IssueError(13);
}

void CAnyType::GenDecode(Scope* S, mlir::MLIRContext* C,
                         const DecodeTarget& target, PathPiece dispatch,
                         PathPiece tag, const CAnyType* cto,
                         mlir::OpBuilder& builder, const mlir::Value& base,
                         const mlir::Value& tagv, const mlir::Value& from,
                         const mlir::Value& to,
                         const mlir::Value& state) const {
  throw IssueError(13);
}

void COutlinedType::GenDecode(Scope* S, mlir::MLIRContext* C,
                              const DecodeTarget& target, PathPiece dispatch,
                              PathPiece tag, const COutlinedType* cto,
                              mlir::OpBuilder& builder, const mlir::Value& base,
                              const mlir::Value& tagv, const mlir::Value& from,
                              const mlir::Value& to,
                              const mlir::Value& state) const {
  throw IssueError(13);
}

void CNamedType::GenDecode(Scope* S, mlir::MLIRContext* C,
                           const DecodeTarget& target, PathPiece dispatch,
                           PathPiece tag, const CNamedType* cto,
                           mlir::OpBuilder& builder, const mlir::Value& base,
                           const mlir::Value& tagv, const mlir::Value& from,
                           const mlir::Value& to,
                           const mlir::Value& state) const {
  assert(false && "CNamedType is only used in sourcegen.");
  return;
}

}  // namespace pj
