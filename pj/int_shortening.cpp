#include <unordered_set>

#include "array_ref.hpp"
#include "defer.hpp"
#include "int_shortening.hpp"

namespace pj {
using namespace mlir;
using namespace types;

std::optional<IntShortening::Split> IntShortening::splitFirstEligibleInt(
    llvm::StringRef name, Type type) {
  if (auto in = type.dyn_cast<IntType>()) {
    return Split{.threshold = 256,
                 .short_type = IntType::get(&ctx_, Int{.width = Bytes(1),
                                                       .alignment = Bytes(1),
                                                       .sign = in->sign}),
                 .original_type = in,
                 .path = PathAttr::fromString(&ctx_, name)};
  }

  if (auto str = type.dyn_cast<StructType>()) {
    std::optional<Split> split;
    uintptr_t i;
    for (i = 0; i < str->fields.size(); ++i) {
      if ((split = splitFirstEligibleInt(str->fields[i].name,
                                         str->fields[i].type))) {
        break;
      }
    }
    if (!split) {
      return {};
    }

    // Make a copy of the current struct with a new name that contains a field
    // of type short_type at position i.
    auto data = Struct(str);

    ArrayRefConverter<StructField> field_conv{data.fields, data.fields.size()};
    field_conv.storage()[i].type = split->short_type;
    data.fields = field_conv.get();

    ArrayRefConverter<llvm::StringRef> name_conv{str.name(), str.name().size()};
    std::string back =
        str.name().back().str() + std::to_string(reinterpret_cast<uintptr_t>(
                                      split->short_type.getAsOpaquePointer()));
    name_conv.storage().back() = back;

    auto short_type = StructType::get(&ctx_, str.domain(), name_conv.get());
    short_type.setTypeData(data);

    return Split{.threshold = split->threshold,
                 .short_type = short_type,
                 .original_type = str,
                 .path = split->path.expand(name)};
  }

  return {};
}

bool IntShortening::shortenInt(VariantType var) {
  ArrayRef<Term> terms = var.terms();
  std::unordered_set<uint64_t> tags;

  for (const Term& t : terms) {
    tags.insert(t.tag);
  }

  for (uintptr_t i = 0; i < terms.size(); ++i) {
    const Term& t = terms[i];
    auto split = splitFirstEligibleInt(t.name, t.type);
    if (!split) {
      continue;
    }

    ArrayRefConverter<Term> term_conv{terms, terms.size()};
    Term& short_term = term_conv.storage().emplace_back(t);
    Term& original_term = term_conv.storage()[i];

    short_term.type = split->short_type;
    original_term.type = split->original_type;

    // Keep the same tag on the original term and set a new tag for the short
    // term.
    short_term.tag = 1;
    while (tags.find(short_term.tag) != tags.end()) {
      ++short_term.tag;
    }

    // Add the VectorSplit attribute to indicate which should be chosen during
    // encoding.
    auto set_attribute = [&split](Term& t, TermAttribute::ShortInt::Type type) {
      ArrayRefConverter<TermAttribute> attributes{t.attributes,
                                                  t.attributes.size()};
      attributes.storage().push_back(
          TermAttribute{.value = TermAttribute::ShortInt{
                            .type = type,
                            .threshold = split->threshold,
                            .path = split->path,
                        }});
      t.attributes = attributes.get();
      return attributes;
    };
    DEFER_DESTRUCTION(
        set_attribute(short_term, TermAttribute::ShortInt::kShort));
    DEFER_DESTRUCTION(
        set_attribute(original_term, TermAttribute::ShortInt::kOriginal));

    // default_term stays the same. If it pointed to a different term, it
    // doesn't matter. If it pointed to the same term, the default will now
    // point to the original version.
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

bool IntShortening::findVariantAndShorten(mlir::Type type) {
  if (auto var = type.dyn_cast<VariantType>()) {
    shortenInt(var);
    return true;
  }

  if (auto str = type.dyn_cast<StructType>()) {
    for (const StructField& f : str->fields) {
      if (findVariantAndShorten(f.type)) {
        return true;
      }
    }
  }

  return false;
}

bool IntShortening::run(Protocol& proto) {
  return findVariantAndShorten(proto.head);
}

}  // namespace pj
