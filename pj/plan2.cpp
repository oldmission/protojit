#include "plan.hpp"
#include "util.hpp"

namespace pj {

using namespace types;

struct PlanningContext {
  mlir::MLIRContext& ctx;
  OutlineVariantType outline;
};

template <typename CB>
auto dispatch(mlir::Type type, CB&& cb) {
  auto id = type.getTypeID();
  if (id == IntType::getTypeID()) {
    return cb(type.dyn_cast<IntType>());
  } else if (id == StructType::getTypeID()) {
    return cb(type.dyn_cast<StructType>());
  } else if (id == InlineVariantType::getTypeID()) {
    return cb(type.dyn_cast<InlineVariantType>());
  } else if (id == ArrayType::getTypeID()) {
    return cb(type.dyn_cast<ArrayType>());
  } else if (id == VectorType::getTypeID()) {
    return cb(type.dyn_cast<VectorType>());
  } else {
    UNREACHABLE();
  }
}

std::optional<PathAttr> into(std::optional<PathAttr> path,
                             llvm::StringRef prefix) {
  if (!path.has_value()) {
    return std::nullopt;
  }
  return path.value().into(prefix);
}

bool matches(std::optional<PathAttr> path) {
  return path.has_value() && path.value().getValue().size() > 0;
}

ValueType plan(PlanningContext& ctx, mlir::Type type,
               std::optional<PathAttr> path);

ValueType plan(PlanningContext& ctx, IntType type,
               std::optional<PathAttr> path) {
  return IntType::get(
      &ctx.ctx,
      Int{.width = type->width, .alignment = Bytes(1), .sign = type->sign});
}

ValueType plan(PlanningContext& ctx, StructType type,
               std::optional<PathAttr> path) {
  Width offset = Bytes(0);
  Width alignment = Bytes(1);

  bool outline_reached = false;
  Width outline_offset_start;

  std::vector<StructField> fields;
  for (const StructField& f : type->fields) {
    auto el = plan(ctx, f.type, into(path, f.name));
    offset = RoundUp(offset, el.head_alignment());
    fields.emplace_back(
        StructField{.type = el, .name = f.name, .offset = offset});
    offset += el.head_size();
    alignment = std::max(alignment, el.head_alignment());

    if (matches(path) && ctx.outline && !outline_reached) {
      outline_reached = true;
      outline_offset_start = offset;
    }
  }

  if (ctx.outline) {
    auto data = OutlineVariant(ctx.outline);
    data.term_offset += offset - outline_offset_start;
    ctx.outline.setTypeData(data);
  }

  auto planned =
      StructType::get(&ctx.ctx, types::TypeDomain::kWire, type.name());
  planned.setTypeData(
      {.fields = fields, .size = offset, .alignment = alignment});
  return planned;
}

ValueType plan(PlanningContext& ctx, InlineVariantType type,
               std::optional<PathAttr> path) {
  Width term_size = Bytes(0);
  Width term_align = Bytes(1);

  llvm::SmallVector<Term, 10> terms;
  for (const Term& term : type->terms) {
    // Outlining variants within variants is not supported, so path is not
    // passed through
    auto planned = plan(ctx, term.type, std::nullopt);
    term_size = std::max(term_size, planned.head_size());
    term_align = RoundUp(term_align, planned.head_alignment());
    terms.push_back({.name = term.name, .type = planned, .tag = term.tag});
  }

  const Width tag_width = compute_tag_width(InlineVariant(type));
  const Width tag_align = Bytes(1);

  bool outline = matches(path);
  if (outline) {
    // Modify the name so that the final identifier starts with a ^ to
    // disambiguate it from the InlineVariant. For example, A::B::C becomes
    // A::B::^C
    assert(type.name().size() > 0);
    std::vector<llvm::StringRef> name{type.name().begin(),
                                      std::prev(type.name().end())};
    std::string last = "^" + type.name().back().str();
    name.push_back(last);
    auto planned = OutlineVariantType::get(&ctx.ctx, types::TypeDomain::kWire,
                                           Name{&name[0], name.size()});

    // term_offset starts out pointing to the end of the OutlineVariant, and is
    // incremented to point to the end of every successive struct it is
    // contained within until it points to the end of the top-level struct.
    planned.setTypeData({.terms = terms,
                         .tag_width = tag_width,
                         .tag_alignment = tag_align,
                         .term_offset = tag_width,
                         .term_alignment = term_align});
    assert(!ctx.outline);
    ctx.outline = planned;
    return planned;
  } else {
    const Width term_offset = Bytes(0);
    const Width tag_offset = RoundUp(term_offset + term_size, tag_align);
    auto planned =
        InlineVariantType::get(&ctx.ctx, types::TypeDomain::kWire, type.name());
    planned.setTypeData({.terms = terms,
                         .term_offset = term_offset,
                         .term_size = term_size,
                         .tag_offset = tag_offset,
                         .tag_width = tag_width,
                         .size = tag_offset + tag_width,
                         .alignment = std::max(term_align, tag_align)});
    return planned;
  }
}

ValueType plan(PlanningContext& ctx, ArrayType type,
               std::optional<PathAttr> path) {
  auto elem = plan(ctx, type->elem, std::nullopt);
  return ArrayType::get(
      &ctx.ctx,
      Array{.elem = elem,
            .length = type->length,
            .elem_size = RoundUp(elem.head_size(), elem.head_alignment()),
            .alignment = elem.head_alignment()});
}

ValueType plan(PlanningContext& ctx, VectorType type,
               std::optional<PathAttr> path) {
  auto elem = plan(ctx, type->elem, std::nullopt);
  const Width el_align = elem.head_alignment();
  assert(el_align.bytes() > 0);
  const Width el_size = RoundUp(elem.head_size(), el_align);

  const intptr_t min_length = type->wire_min_length;

  const Width length_offset = Bytes(0);
  Width length_size = Bytes(8);
  if (type->max_length >= 0) {
    length_size = Bytes(RoundUpToPowerOfTwo(
        DivideUp<intptr_t>(llvm::Log2_64_Ceil(type->max_length), kByte)));
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
  const Width partial_payload_size = Bytes(el_size.bytes() * min_length);

  const Width size =
      std::max(inline_payload_offset + inline_payload_size,
               (ppl_count > 0) ? partial_payload_offset + partial_payload_size
                               : ref_offset + ref_size);
  const Width alignment =
      (min_length > 0 || ppl_count > 0) ? el_align : Bytes(1);

  return VectorType::get(
      &ctx.ctx, Vector{.elem = elem,
                       .min_length = min_length,
                       .max_length = type->max_length,
                       .wire_min_length = min_length,
                       .ppl_count = ppl_count,
                       .length_offset = length_offset,
                       .length_size = length_size,
                       .ref_offset = ref_offset,
                       .ref_size = ref_size,
                       .reference_mode = pj::types::Vector::kOffset,
                       .inline_payload_offset = inline_payload_offset,
                       .inline_payload_size = inline_payload_size,
                       .partial_payload_offset = partial_payload_offset,
                       .partial_payload_size = partial_payload_size,
                       .size = size,
                       .alignment = alignment,
                       .outlined_payload_alignment = el_align});
}

ValueType plan(PlanningContext& ctx, mlir::Type type,
               std::optional<PathAttr> path) {
  return dispatch(type, [&ctx, path](auto t) { return plan(ctx, t, path); });
}

ValueType plan(mlir::MLIRContext& mlir_ctx, mlir::Type type,
               std::optional<PathAttr> path) {
  PlanningContext ctx{.ctx = mlir_ctx, .outline = OutlineVariantType{}};

  auto planned =
      dispatch(type, [&ctx, path](auto t) { return plan(ctx, t, path); });

  if (ctx.outline) {
    // Update the outlined variant's term offset to the correct alignment
    auto data = OutlineVariant(ctx.outline);
    data.term_offset = RoundUp(data.term_offset, data.term_alignment);
    ctx.outline.setTypeData(data);
  }

  return planned;
}

}  // namespace pj
