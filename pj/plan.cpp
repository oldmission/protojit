#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <llvm/Support/MathExtras.h>

#include "concrete_types.hpp"
#include "protocol.hpp"
#include "tag.hpp"
#include "util.hpp"

namespace pj {

using std::pair;

const CType* CArrayType::Plan(Scope& scope, const ProtoParams& params,
                              PathPiece tag) const {
  return this;
}

const CType* CListType::Plan(Scope& scope, const ProtoParams& params,
                             PathPiece tag) const {
  assert(IsEmptyTag(tag));
  auto inner = el->Plan(scope, params, tag);

  const Width len_offset = Bits(0);

  auto len_size = Width::None();
  if (abs()->max_len == kNone) {
    len_size = Bytes(8);
  } else {
    len_size = Bytes(RoundUpToPowerOfTwo(
        DivideUp<intptr_t>(llvm::Log2_64_Ceil(abs()->max_len), kByte)));
    assert(len_size.bytes() <= 8);
  }

  const Width el_size = RoundUp(inner->total_size(), inner->alignment());

  const Width ref_offset = len_offset + len_size;

  auto ref_size = Width::None();
  if (params.max_size.IsNotNone()) {
    ref_size = Bytes(
        DivideUp<intptr_t>(llvm::Log2_64_Ceil(params.max_size.bytes()), kByte));
  } else {
    ref_size = Bytes(8);
  }

  const Width partial_payload_offset =
      RoundUp(ref_offset + ref_size, inner->alignment());

  const Width full_payload_offset =
      RoundUp(len_offset + len_size, inner->alignment());

  const auto full_payload_size = Bytes(el_size.bytes() * abs()->min_len);

  const auto partial_payload_count =
      DivideDown(std::max(Bits(0), full_payload_offset + full_payload_size -
                                       partial_payload_offset),
                 el_size);

  const Width total_size =
      full_payload_size.IsPos()
          ? std::max(full_payload_offset + full_payload_size,
                     partial_payload_offset + el_size * partial_payload_count)
          : ref_offset + ref_size;

  const auto alignment = full_payload_size.IsPos()
                             ? std::max(Bytes(1), inner->alignment())
                             : Bytes(1);

  const auto* const ctype = new (scope)
      CListType(abs(), alignment, total_size, inner, ref_offset, ref_size,
                partial_payload_offset, partial_payload_count,
                full_payload_offset, abs()->min_len, len_offset, len_size);

  return ctype;
}

const CType* CIntType::Plan(Scope& scope, const ProtoParams& params,
                            PathPiece tag) const {
  auto* new_type = new (scope) CIntType(abs(), Bytes(1), total_size());
  return new_type;
}

const CType* CAnyType::Plan(Scope& scope, const ProtoParams& params,
                            PathPiece tag) const {
  return this;
}

const CType* CStructType::Plan(Scope& scope, const ProtoParams& params,
                               PathPiece tag) const {
  std::vector<std::pair<std::string, CStructField>> orig_fields;

  std::string tagged_field = "";
  if (tag.begin != tag.end) {
    tagged_field = *tag.begin;
  }

  for (auto& [n, f] : fields) {
    if (n != tagged_field) {
      orig_fields.emplace_back(n, f);
    }
  }

  // Try to keep the fields in their original offset order.
  std::sort(orig_fields.begin(), orig_fields.end(),
            [&](const auto& x, const auto& y) { return x.first < y.first; });

  std::map<std::string, CStructField> new_fields;
  Width offset = Bytes(0);
  Width alignment = Bytes(1);
  for (auto& [n, f] : orig_fields) {
    auto nt = f.type->Plan(scope, params, {});
    offset = RoundUp(offset, nt->alignment());
    new_fields.emplace(n, CStructField{.offset = offset, .type = nt});
    offset += nt->total_size();
    alignment = std::max(alignment, nt->alignment());
  }

  // Add in the tagged field, if we have one.
  Width total_size = offset;
  if (not tagged_field.empty()) {
    auto tt = fields.at(tagged_field)
                  .type->Plan(scope, params, Narrow(tag, tagged_field));
    offset = RoundUp(offset, tt->alignment());
    alignment = std::max(alignment, tt->alignment());

    if (tt->total_size().IsNone()) {
      total_size = Width::None();
    } else {
      total_size = offset + tt->total_size();
    }

    new_fields.emplace(tagged_field,
                       CStructField{.offset = offset, .type = tt});
  }

  // TODO(27): nested struct tags
  return new (scope)
      CStructType(abs(), alignment, total_size, std::move(new_fields));
}

const CType* CVariantType::Plan(Scope& scope, const ProtoParams& params,
                                PathPiece tag) const {
  assert(IsEmptyTag(tag) || IsDotTag(tag));

  Width total_size = Bytes(0);
  Width alignment = Bytes(1);

  std::map<std::string, CTerm> planned_types;
  for (auto& [name, t] : this->terms) {
    auto pt = t.type->Plan(scope, params, {});
    alignment = RoundUp(alignment, pt->alignment());
    total_size = std::max(total_size, pt->total_size());
    planned_types.emplace(name, CTerm{.tag = t.tag, .type = pt});
  }

  Width tag_offset = Width::None();
  if (IsEmptyTag(tag)) {
    tag_offset = total_size;
    total_size += tag_size;
  } else {
    total_size = Width::None();
  }

  return new (scope) CVariantType{
      abs(),
      alignment,
      total_size,
      std::move(planned_types),
      /*term_offset=*/Bits(0),
      /*term_size=*/total_size,
      /*tag_offset=*/tag_offset,
      /*tag_size=*/tag_size,
  };
}

const CType* CNamedType::Plan(Scope& scope, const ProtoParams& params,
                              PathPiece path) const {
  assert(false && "CNamedType is only used in sourcegen.\n");
  return {};
}

const CType* AIntType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  // TODO: validate
  assert(len.bytes() <= kMaxCppIntSize);
  assert(__builtin_popcount(len.bytes()) == 1);
  return new (S) CIntType(this, len, len);
}

const CType* AAnyType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  // SAMIR_TODO: NYI
  return nullptr;
}

const CType* AVariantType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  return PlanWithTags(S, memo, {});
}

const CType* AVariantType::PlanWithTags(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo,
    std::map<std::string, uint8_t> explicit_tags) const {
  std::map<std::string, CVariantType::CTerm> cterms;
  std::unordered_set<uint8_t> reserved_tags;
  for (auto& [_, t] : explicit_tags) reserved_tags.emplace(t);

  auto tag_offset = Bytes(0);
  auto max_align = Bytes(1);
  intptr_t next_tag = 0;
  intptr_t max_tag = 0;

  for (auto& [name, term] : terms) {
    if (!term) continue;
    auto* ct = term->Plan(S, memo);
    tag_offset = std::max(tag_offset, ct->total_size());
    max_align = std::max(max_align, ct->alignment());
    intptr_t tag;
    if (auto it = explicit_tags.find(name); it != explicit_tags.end()) {
      tag = it->second;
    } else {
      while (reserved_tags.count(next_tag)) next_tag++;
      tag = next_tag++;
    }
    cterms.emplace(name, CVariantType::CTerm{.tag = tag, .type = ct});
    max_tag = std::max(max_tag, tag);
  }

  // SAMIR_TODO
  assert(max_tag < (1 << 8));

  // Align up the size.
  const auto total_size = RoundUp(tag_offset + Bytes(1), max_align);

  return new (S) CVariantType{
      this,
      max_align,
      total_size,
      std::move(cterms),
      /*term_offset=*/Bits(0),
      /*term_size=*/tag_offset,
      /*tag_offset=*/tag_offset,
      /*tag_size=*/Bytes(1),
  };
}

const CType* AStructType::PlanWithFieldOrder(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo,
    const std::vector<std::string>& field_order) const {
  std::vector<std::pair<std::string, const CType*>> ctypes;

  if (field_order.empty()) {
    for (auto& [n, t] : fields) {
      ctypes.emplace_back(n, t->Plan(S, memo));
    }
  } else {
    for (auto& n : field_order) {
      ctypes.emplace_back(n, fields.at(n)->Plan(S, memo));
    }
  }

  if (field_order.empty()) {
    // Sort fields by decreasing alignment. This minimizes padding space.
    std::sort(ctypes.begin(), ctypes.end(), [&](const auto& l, const auto& r) {
      return l.second->alignment() > r.second->alignment();
    });
  }

  auto offset = Bytes(0);
  auto alignment = Bytes(1);
  std::map<std::string, CStructType::CStructField> cfields;
  for (auto& [n, ct] : ctypes) {
    alignment = std::max(ct->alignment(), alignment);
    offset = RoundUp(offset, ct->alignment());
    cfields.emplace(n, CStructType::CStructField{
                           .offset = offset,
                           .type = ct,
                       });
    offset += ct->total_size();
  }

  if (offset.IsZero()) {
    assert(cfields.empty());

    // In C, an empty struct has size = 1 byte.
    return new (S) CStructType(this, Bytes(1), Bytes(1), std::move(cfields));
  }

  // In C, size is always a multiple of alignment.
  offset = RoundUp(offset, alignment);

  return new (S) CStructType(this, alignment, offset, std::move(cfields));
}

const CType* AStructType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  return PlanWithFieldOrder(S, memo, {});
}

const CType* ANamedType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  return new (S) CNamedType(this, named->PlanMemo(S, memo));
}

const CType* AArrayType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  auto elc = el->Plan(S, memo);
  assert(elc->total_size().IsAlignedTo(elc->alignment()));
  return new (S)
      CArrayType(this, elc, elc->alignment(), elc->total_size() * length);
}

const CType* AOutlinedType::PlanMemo(
    Scope*, std::unordered_map<const AType*, const CType*>& memo) const {
  throw IssueError(-1);
  return nullptr;
}

const CType* AListType::PlanMemo(
    Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
  return new CListType{
      /*abs=*/this,
      /*alignment=*/Bytes(8),
      /*total_size=*/Bytes(16),
      /*el=*/el->Plan(S, memo),
      /*ref_offset=*/Bytes(8),
      /*ref_size=*/Bytes(8),
      /*partial_payload_offset=*/Bytes(8),
      /*partial_payload_count=*/kNone,
      /*full_payload_offset=*/Bytes(8),
      /*full_payload_count=*/0,
      /*len_offset=*/Bytes(0),
      /*len_size=*/Bytes(8),
  };
}

}  // namespace pj
