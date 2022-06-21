#include <unordered_set>

#include "defer.hpp"
#include "span.hpp"
#include "vector_hoisting.hpp"

namespace pj {
using namespace mlir;
using namespace types;

std::optional<VectorHoisting::Split> VectorHoisting::splitFirstEligibleVector(
    Type type) {
  if (auto vec = type.dyn_cast<VectorType>()) {
    if (vec->wire_min_length == 0 || vec->wire_min_length == vec->max_length) {
      return {};
    }
    return Split{
        .inline_length = vec->wire_min_length,
        .short_type = VectorType::get(
            &ctx_,
            Vector{.elem = vec->elem,
                   .max_length = static_cast<int64_t>(vec->wire_min_length),
                   .wire_min_length = vec->wire_min_length}),
        .long_type =
            VectorType::get(&ctx_, Vector{.elem = vec->elem,
                                          .max_length = vec->max_length,
                                          .wire_min_length = 0}),
        .path = PathAttr::fromString(&ctx_, "_")};
  }

  if (auto str = type.dyn_cast<StructType>()) {
    std::optional<Split> split;
    uintptr_t i;
    for (i = 0; i < str->fields.size(); ++i) {
      if ((split = splitFirstEligibleVector(str->fields[i].type))) {
        break;
      }
    }
    if (!split) {
      return {};
    }

    // Gets a copy of the current struct with a new name that contains a field
    // of type inner at position i instead of the existing type.
    auto get_struct = [&](ValueType inner) {
      auto data = Struct(str);

      SpanConverter<StructField> field_conv{data.fields, data.fields.size()};
      field_conv.storage()[i].type = inner;
      data.fields = field_conv.get();

      SpanConverter<llvm::StringRef> name_conv{str.name(), str.name().size()};
      std::string back =
          str.name().back().str() + std::to_string(reinterpret_cast<uintptr_t>(
                                        inner.getAsOpaquePointer()));
      name_conv.storage().back() = back;

      auto outer = StructType::get(&ctx_, str.domain(), name_conv.get());
      outer.setTypeData(data);
      return outer;
    };

    return Split{.inline_length = split->inline_length,
                 .short_type = get_struct(split->short_type),
                 .long_type = get_struct(split->long_type),
                 .path = split->path.expand(str->fields[i].name)};
  }

  return {};
}

bool VectorHoisting::hoistVectors(VariantType var) {
  Span<Term> terms = var.terms();
  std::unordered_set<uint64_t> tags;

  for (const Term& t : terms) {
    tags.insert(t.tag);
  }

  for (uintptr_t i = 0; i < terms.size(); ++i) {
    const Term& t = terms[i];
    auto split = splitFirstEligibleVector(t.type);
    if (!split) {
      continue;
    }

    SpanConverter<Term> term_conv{terms, terms.size()};
    Term& short_term = term_conv.storage().emplace_back(t);
    Term& long_term = term_conv.storage()[i];

    short_term.type = split->short_type;
    long_term.type = split->long_type;

    // Keep the same tag on the long term and set a new tag for the short term.
    short_term.tag = 1;
    while (tags.find(short_term.tag) != tags.end()) {
      ++short_term.tag;
    }

    // Add the VectorSplit attribute to indicate which should be chosen during
    // encoding.
    auto set_attribute = [&split](Term& t,
                                  TermAttribute::VectorSplit::Type type) {
      SpanConverter<TermAttribute> attributes{t.attributes,
                                              t.attributes.size()};
      attributes.storage().push_back(TermAttribute{
          .value = TermAttribute::VectorSplit{
              .type = type,
              .inline_length = split->inline_length,
              .path = split->path,
              .is_default = type == TermAttribute::VectorSplit::kOutline,
          }});
      t.attributes = attributes.get();
      return attributes;
    };
    DEFER_DESTRUCTION(
        set_attribute(short_term, TermAttribute::VectorSplit::kInline));
    DEFER_DESTRUCTION(
        set_attribute(long_term, TermAttribute::VectorSplit::kOutline));

    if (auto as_inline = var.dyn_cast<InlineVariantType>()) {
      auto data = InlineVariant(as_inline);
      data.terms = term_conv.get();
      as_inline.setTypeData(data);
    } else {
      auto as_outline = var.cast<OutlineVariantType>();
      auto data = OutlineVariant(as_outline);
      data.terms = term_conv.get();
      as_outline.setTypeData(data);
    }

    return true;
  }

  return false;
}

bool VectorHoisting::findVariantAndHoist(mlir::Type type) {
  if (auto var = type.dyn_cast<VariantType>()) {
    while (hoistVectors(var)) {
      // Hoist vectors until there are none left to hoist.
    }
    return true;
  }

  if (auto str = type.dyn_cast<StructType>()) {
    for (const StructField& f : str->fields) {
      if (findVariantAndHoist(f.type)) {
        return true;
      }
    }
  }

  return false;
}

bool VectorHoisting::run(Protocol& proto) {
  return findVariantAndHoist(proto.head);
}

}  // namespace pj
