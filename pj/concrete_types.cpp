#include "concrete_types.hpp"

#include "ir.hpp"
#include "tag.hpp"

namespace pj {
CType::~CType() {}

CVariantType::CVariantType(const AVariantType* abs, Width alignment,
                           Width total_size, decltype(terms)&& terms,
                           Width term_offset, Width term_size, Width tag_offset,
                           Width tag_size)
    : CType(abs, alignment, total_size),
      terms(terms),
      term_offset(term_offset),
      term_size(term_size),
      tag_offset(tag_offset),
      tag_size(tag_size) {
  // TODO(8): move these checks to validate
  assert(term_offset.bits() != kNone);
  assert(tag_size.IsNotNone());
  assert(tag_offset.IsNone() ||
         (total_size.IsNotNone() && term_size.IsNotNone()));
}

CListType::CListType(const AListType* abs, Width alignment, Width total_size,
                     const CType* el, Width ref_offset, Width ref_size,
                     Width partial_payload_offset,
                     intptr_t partial_payload_count, Width full_payload_offset,
                     intptr_t full_payload_count, Width len_offset,
                     Width len_size)
    : CType(abs, alignment, total_size),
      el(el),
      len_offset(len_offset),
      len_size(len_size),
      ref_offset(ref_offset),
      ref_size(ref_size),
      partial_payload_count(partial_payload_count),
      partial_payload_offset(partial_payload_offset),
      full_payload_offset(full_payload_offset),
      full_payload_count(full_payload_count) {}

CAnyType::CAnyType(AAnyType* abs, Width alignment, Width total_size,
                   intptr_t data_offset, intptr_t data_size,
                   intptr_t type_offset, intptr_t type_size,
                   intptr_t type_id_offset, intptr_t type_id_size)
    : CType(abs, alignment, total_size),
      data_offset(data_offset),
      data_size(data_size),
      type_offset(type_offset),
      type_size(type_size),
      type_id_offset(type_id_offset),
      type_id_size(type_id_size) {}

Width CVariantType::ImpliedSize(const CType* from, PathPiece path,
                                PathPiece tag) const {
  if (total_size().IsNotNone()) return total_size();

  if (IsEmptyTag(path)) {
    return !from or !from->IsVariant() ? term_offset : total_size();
  }

  // TODO(8): move into validate
  assert(IsDotTag(tag));

  if (*path.begin == "undef") return term_offset;

  assert(terms.count(*path.begin));
  assert(term_size.IsNone());

  const CType* from_inner = nullptr;
  if (from and from->IsVariant()) {
    if (auto it = from->AsVariant()->terms.find(*path.begin);
        it != from->AsVariant()->terms.end()) {
      from_inner = it->second.type;
    }
  }

  return term_offset +
         terms.at(*path.begin)
             .type->ImpliedSize(from_inner, Tail(path), Tail(tag));
}

const CVariantType* CVariantType::RemoveTag(Scope* scope) const {
  assert(term_offset.bits() == 0);
  return new (scope)
      CVariantType(abs(), alignment(), Width::None(), decltype(terms)(terms),
                   term_offset, Width::None(), Width::None(), tag_size);
}

CNamedType::~CNamedType() {}

bool CVariantType::MatchesExactlyAsEnum(const CVariantType* other) const {
  if (tag_size != other->tag_size) return false;
  if (term_size.bits() != 0 || other->term_size.bits() != 0) return false;
  if (terms.size() != other->terms.size()) return false;
  for (auto& [n, t] : terms) {
    auto i = other->terms.find(n);
    if (i == other->terms.end()) return false;
    if (t.tag != i->second.tag) return false;
  }
  return true;
}

mlir::Type CVariantType::GetTagAsIntegerType(Scope* S,
                                             mlir::MLIRContext* C) const {
  assert(tag_size.bits() > 0);
  return (new (S) CIntType(
              new (S) AIntType(tag_size, AIntType::Conversion::kUnsigned),
              Bytes(1), tag_size))
      ->toIR(C);
}

Width CStructType::ImpliedSize(const CType* from, PathPiece path,
                               PathPiece tag) const {
  if (total_size().IsNotNone()) return total_size();

  assert(not IsEmptyTag(tag));
  assert(not IsDotTag(tag));
  assert(fields.count(*tag.begin));

  const auto& field = fields.at(*tag.begin);

  const CType* inner_field = nullptr;
  if (from and from->IsStruct()) {
    if (auto it = from->AsStruct()->fields.find(*tag.begin);
        it != from->AsStruct()->fields.end()) {
      inner_field = it->second.type;
    }
  }

  const auto fsize =
      field.type->ImpliedSize(inner_field, Narrow(path, *tag.begin), Tail(tag));
  return fsize.IsNone() ? Width::None() : field.offset + fsize;
}

}  // namespace pj
