#include "wire_layout.hpp"

namespace pj {
using namespace mlir;
using namespace types;

ValueType WireLayout::visit(IntType type) {
  return IntType::get(
      &ctx_,
      Int{.width = type->width, .alignment = Bytes(1), .sign = type->sign});
}

ValueType WireLayout::visit(UnitType type) { return type; }

ValueType WireLayout::visit(StructType type) {
  Width offset = Bytes(0);
  Width alignment = Bytes(1);

  std::vector<StructField> fields;
  for (const StructField& f : type->fields) {
    auto el = visit(f.type);
    offset = RoundUp(offset, el.headAlignment());
    fields.emplace_back(
        StructField{.type = el, .name = f.name, .offset = offset});
    offset += el.headSize();
    alignment = std::max(alignment, el.headAlignment());
  }

  auto packed = StructType::get(&ctx_, wire_domain_, type.name());
  packed.setTypeData(
      {.fields = fields, .size = offset, .alignment = alignment});
  return packed;
}

template <typename Variant>
ValueType WireLayout::visitVariant(Variant type) {
  Width term_size = Bytes(0);
  Width term_align = Bytes(1);

  llvm::SmallVector<Term, 10> terms;
  for (const Term& term : type->terms) {
    auto packed = visit(term.type);
    term_size = std::max(term_size, packed.headSize());
    term_align = RoundUp(term_align, packed.headAlignment());

    auto& new_term = terms.emplace_back(term);
    new_term.type = packed;
  }

  const Width tag_width = compute_tag_width(type);
  const Width tag_align = Bytes(1);

  if constexpr (std::is_same_v<Variant, OutlineVariantType>) {
    auto packed = OutlineVariantType::get(&ctx_, wire_domain_, type.name());
    // term_offset will be set in OutlineVariantOffsetGeneration.
    packed.setTypeData({.terms = terms,
                        .tag_width = tag_width,
                        .tag_alignment = tag_align,
                        .term_offset = Bytes(0),
                        .term_alignment = term_align});
    return packed;
  } else {
    const Width term_offset = Bytes(0);
    const Width tag_offset = RoundUp(term_offset + term_size, tag_align);
    auto packed = InlineVariantType::get(&ctx_, wire_domain_, type.name());
    packed.setTypeData({.terms = terms,
                        .term_offset = term_offset,
                        .term_size = term_size,
                        .tag_offset = tag_offset,
                        .tag_width = tag_width,
                        .size = tag_offset + tag_width,
                        .alignment = std::max(term_align, tag_align)});
    return packed;
  }
}

ValueType WireLayout::visit(InlineVariantType type) {
  return visitVariant(type);
}

ValueType WireLayout::visit(OutlineVariantType type) {
  return visitVariant(type);
}

ValueType WireLayout::visit(ArrayType type) {
  auto elem = visit(type->elem);
  return ArrayType::get(
      &ctx_, Array{.elem = elem,
                   .length = type->length,
                   .elem_size = RoundUp(elem.headSize(), elem.headAlignment()),
                   .alignment = elem.headAlignment()});
}

ValueType WireLayout::visit(VectorType type) {
  auto elem = visit(type->elem);
  const Width el_align = elem.headAlignment();
  assert(el_align.bytes() > 0);
  const Width el_size = RoundUp(elem.headSize(), el_align);

  const uint64_t min_length = type->wire_min_length;

  const Width length_offset = Bytes(0);
  Width length_size = Bytes(8);
  if (type->max_length >= 0) {
    length_size = Bytes(RoundUpToPowerOfTwo(
        DivideUp<intptr_t>(llvm::Log2_64_Ceil(type->max_length + 1), kByte)));
    assert(length_size.bytes() <= 8);
  }

  const Width ref_offset = length_offset + length_size;
  Width ref_size = Bytes(8);
  // TODO: limit the ref_size based on the maximum buffer size

  const Width inline_payload_offset =
      RoundUp(length_offset + length_size, el_align);
  const Width inline_payload_size = Bytes(el_size.bytes() * min_length);

  const Width partial_payload_offset = RoundUp(ref_offset + ref_size, el_align);
  const intptr_t ppl_count =
      DivideDown(std::max(Bits(0), inline_payload_offset + inline_payload_size -
                                       partial_payload_offset),
                 el_size);
  const Width partial_payload_size = Bytes(el_size.bytes() * ppl_count);

  const Width size =
      std::max(inline_payload_offset + inline_payload_size,
               (ppl_count > 0) ? partial_payload_offset + partial_payload_size
                               : ref_offset + ref_size);
  const Width alignment =
      (min_length > 0 || ppl_count > 0) ? el_align : Bytes(1);

  return VectorType::get(
      &ctx_, Vector{.elem = elem,
                    .min_length = min_length,
                    .max_length = type->max_length,
                    .wire_min_length = min_length,
                    .ppl_count = ppl_count,
                    .length_offset = length_offset,
                    .length_size = length_size,
                    .ref_offset = ref_offset,
                    .ref_size = ref_size,
                    .reference_mode = ReferenceMode::kOffset,
                    .inline_payload_offset = inline_payload_offset,
                    .inline_payload_size = inline_payload_size,
                    .partial_payload_offset = partial_payload_offset,
                    .partial_payload_size = partial_payload_size,
                    .size = size,
                    .alignment = alignment,
                    .outlined_payload_alignment = el_align});
}

ValueType WireLayout::visit(ProtocolType type) {
  // ProtocolType is not being used in planning.
  UNREACHABLE();
}

ValueType WireLayout::visit(AnyType type) {
  // TODO: AnyType is not yet implemented
  UNREACHABLE();
}

ValueType WireLayout::visit(mlir::Type type) {
  auto id = type.getTypeID();
#define TRY_VISIT(TYPE) \
  if (id == TYPE::getTypeID()) return visit(type.cast<TYPE>());
  FOR_EACH_VALUE_TYPE(TRY_VISIT)
#undef TRY_VISIT
  UNREACHABLE();
}

bool WireLayout::run(Protocol& proto) {
  proto.head = visit(proto.head);
  return true;
}

}  // namespace pj
