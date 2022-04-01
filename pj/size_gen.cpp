#include "concrete_types.hpp"
#include "defer.hpp"
#include "ir.hpp"
#include "tag.hpp"

namespace pj {
using namespace ir;

mlir::Value CIntType::GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                              const CType* cto, mlir::OpBuilder& _,
                              const mlir::Value& source) const {
  return GetIndexConstant(source.getLoc(), _, 0);
}

mlir::Value GenTermSize(mlir::MLIRContext* C, Scope* S, const std::string& term,
                        PathPiece path, const CVariantType* from,
                        const CType* from_type, const CVariantType* to,
                        mlir::OpBuilder& _, const mlir::Value& source) {
  const auto L = source.getLoc();

  if (to->tag_offset.IsNone()) {
    // SAMIR_TODO: Make sure total_size is set to none even when it might be
    // known for an outer-tagged variant.
    assert(to->total_size().IsNone());

    // Target tag is external. We thus have to factor in internal size
    // as well as the external size.

    if (auto it = to->terms.find(term); it != to->terms.end()) {
      // Source term exists in target.
      // return term internal size + term external size
      auto term = _.create<ProjectOp>(L, from_type->toIR(C), source,
                                      from->term_offset.bytes());
      const auto extern_size =
          from_type->GenSize(S, C, path, it->second.type, _, term);
      assert(it->second.type->total_size().IsNotNone());
      const auto intern_size =
          GetIndexConstant(L, _, it->second.type->total_size().bytes());
      return _.create<AddIOp>(L, extern_size, intern_size);
    } else {
      // Term body has no size, and the tag is not included.
      return GetIndexConstant(L, _, 0);
    }
  } else {
    // Internal size should be factored into enclosing call.
    assert(to->total_size().IsNotNone());

    if (auto it = to->terms.find(term); it != to->terms.end()) {
      auto term = _.create<ProjectOp>(L, from_type->toIR(C), source,
                                      from->term_offset.bytes());
      return from_type->GenSize(S, C, path, it->second.type, _, term);
    } else {
      // Term body has no external size, and internal size of the variant
      // is factored in by the enclosing call.
      return GetIndexConstant(L, _, 0);
    }
  }
}

mlir::Value CVariantType::GenSize(Scope* S, mlir::MLIRContext* C,
                                  PathPiece path, const CType* cto_,
                                  mlir::OpBuilder& _,
                                  const mlir::Value& source) const {
  auto L = source.getLoc();
  auto* cto = cto_->AsVariant();

  // TODO: verify this
  assert(cto->tag_offset.IsNone() or cto->total_size().IsNotNone());

  if (!IsEmptyTag(path)) {
    // Source term is known.
    auto term = *path.begin;
    const auto from_type = terms.at(term).type;
    assert(terms.count(term));
    return GenTermSize(C, S, *path.begin, Tail(path), this, from_type, cto, _,
                       source);
  } else {
    // Source term is not known, dispatch on it.
    auto match = _.create<MatchVariantOp>(
        L, TypeRange{mlir::IndexType::get(_.getContext())}, source);
    Region* body = &match.body();

    DEFER(_.setInsertionPointAfter(match));

    auto entry = new Block();
    body->push_back(entry);

    llvm::SmallVector<Block*, 8> successors;
    llvm::SmallVector<uint64_t, 8> tags;

    auto* undef = new Block();
    body->push_back(undef);
    _.setInsertionPointToStart(undef);
    auto sz = GenTermSize(C, S, "undef", {}, this, /*S->CUnit()*/ nullptr, cto,
                          _, source);
    _.create<RetOp>(L, sz);
    successors.push_back(undef);
    tags.push_back(kUndefTag);

    for (auto& [name, term] : terms) {
      if (auto it = cto->terms.find(name); it != cto->terms.end()) {
        auto* const term_block = new Block();
        _.setInsertionPointToStart(term_block);
        auto size =
            GenTermSize(C, S, name, {}, this, term.type, cto, _, source);
        _.create<RetOp>(L, size);
        successors.push_back(term_block);
        tags.push_back(term.tag);
      } else {
        successors.push_back(undef);
        tags.push_back(kUndefTag);
      }
    }

    _.setInsertionPointToEnd(entry);
    auto match_tag = _.create<DTagOp>(L, _.getIntegerType(tag_size.bits()),
                                      source, tag_offset);
    _.create<BTagOp>(L, match_tag, llvm::makeArrayRef(tags), successors);

    return match.getResult(0);
  }
}

mlir::Value CNamedType::GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                                const CType* cto, mlir::OpBuilder& _,
                                const mlir::Value& source) const {
  return named->GenSize(S, C, path, cto, _, source);
}

mlir::Value CStructType::GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                                 const CType* cto, mlir::OpBuilder& _,
                                 const mlir::Value& source) const {
  auto L = source.getLoc();

  auto sstr = _.create<SStrOp>(L, mlir::IndexType::get(C), source);

  DEFER(_.setInsertionPointAfter(sstr));

  auto entry = new Block();
  sstr.body().push_back(entry);
  _.setInsertionPointToStart(entry);

  auto total = GetIndexConstant(L, _, 0);
  for (auto& [name, t_field] : cto->AsStruct()->fields) {
    if (auto it = fields.find(name); it != fields.end()) {
      auto& [n, f_field] = *it;
      auto inner = _.create<ProjectOp>(source.getLoc(), f_field.type->toIR(C),
                                       source, f_field.offset.bytes());
      // SAMIR_TODO: fix path
      auto inner_size =
          f_field.type->GenSize(S, C, path, t_field.type, _, inner);
      total = _.create<AddIOp>(L, inner_size, total);
    }
  }

  _.create<RetOp>(L, total);

  return sstr;
}

mlir::Value CArrayType::GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                                const CType* cto, mlir::OpBuilder& _,
                                const mlir::Value& source) const {
  // SAMIR_TODO
  throw IssueError(-1);
}

mlir::Value CListType::GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                               const CType* cto, mlir::OpBuilder& _,
                               const mlir::Value& source) const {
  auto L = source.getLoc();

  auto op = _.create<SListOp>(L, mlir::IndexType::get(C), source,
                              TypeAttr::get(cto->toIR(C)));
  DEFER(_.setInsertionPointAfter(op));

  auto entry = new Block();
  op.body().push_back(entry);
  _.setInsertionPointToStart(entry);

  // SAMIR_TODO: fix path
  auto arg = entry->addArgument(el->toIR(C));
  auto inner = el->GenSize(S, C, path, cto->AsList()->el, _, arg);
  _.create<RetOp>(L, inner);

  return op;
}

mlir::Value CAnyType::GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                              const CType* cto, mlir::OpBuilder& _,
                              const mlir::Value& source) const {
  // SAMIR_TODO
  throw IssueError(-1);
}

};  // namespace pj
