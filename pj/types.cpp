#include <sstream>

#include "types.hpp"
#include "util.hpp"

namespace pj {
namespace types {

Struct type_intern(mlir::TypeStorageAllocator& allocator, const Struct& key) {
  Struct s;

  auto fields = reinterpret_cast<StructField*>(allocator.allocate(
      sizeof(StructField) * key.fields.size(), alignof(StructField)));
  s.size = key.size;
  s.alignment = key.alignment;

  s.has_max_size = true;
  for (uintptr_t i = 0; i < key.fields.size(); ++i) {
    s.has_max_size = s.has_max_size && key.fields[i].type.hasMaxSize();
    fields[i] = StructField{
        .type = key.fields[i].type,
        .name = allocator.copyInto(key.fields[i].name),
        .offset = key.fields[i].offset,
    };

    if (key.fields[i].type.isa<OutlineVariantType>()) {
      assert(!s.outline_variant);
      s.outline_variant = key.fields[i].type;
    } else if (auto child = key.fields[i].type.dyn_cast<StructType>()) {
      if (child->outline_variant) {
        assert(!s.outline_variant);
        s.outline_variant = child->outline_variant;
      }
    }
  }

  s.fields = {fields, key.fields.size()};
  return s;
}

std::string TermAttribute::toString() const {
  std::ostringstream sstr;
  std::visit(
      overloaded{[&](const Undef& undef) {
                   sstr << "UNDEF" << (undef.is_default ? " default" : "");
                 },
                 [&](const VectorSplit& split) {
                   sstr << "VECTOR_SPLIT "
                        << (split.type == VectorSplit::kInline ? "inline, "
                                                               : "outline, ")
                        << split.inline_length << ", " << split.path.toString();
                 }},
      value);
  return sstr.str();
}

std::string Term::toString() const {
  std::ostringstream sstr;
  sstr << std::string_view(name) << " = " << tag;
  if (attributes.size() > 0) {
    sstr << " [";
    bool first = true;
    for (const TermAttribute& attr : attributes) {
      if (!first) {
        sstr << " | ";
      }
      first = false;
      sstr << attr.toString();
    }
    sstr << "]";
  }
  return sstr.str();
}

template <typename T, typename Eq = std::equal_to<T>>
bool isBinaryCompatible(Span<T> a, Span<T> b, Eq&& eq = {}) {
  if (a.size() != b.size()) return false;

  std::vector<T> vec_a{a.begin(), a.end()};
  std::vector<T> vec_b{b.begin(), b.end()};

  for (const T& a : vec_a) {
    auto it = std::find_if(vec_b.begin(), vec_b.end(),
                           [&](const T& b) { return eq(a, b); });
    if (it == vec_b.end()) {
      return false;
    }
    vec_b.erase(it);
  }

  return true;
}

bool Struct::isBinaryCompatibleWith(const Struct& other) const {
  if (!isBinaryCompatible(fields, other.fields,
                          [](const StructField& a, const StructField& b) {
                            return a.type.isBinaryCompatibleWith(b.type) &&
                                   a.name == b.name && a.offset == b.offset;
                          })) {
    return false;
  }

  return size == other.size && alignment == other.alignment;
}

bool isBinaryCompatible(Span<Term> a, Span<Term> b) {
  return isBinaryCompatible(a, b, [](const Term& a, const Term& b) {
    return a.name == b.name && a.type.isBinaryCompatibleWith(a.type) &&
           a.tag == b.tag && isBinaryCompatible(a.attributes, b.attributes);
  });
}

bool InlineVariant::isBinaryCompatibleWith(const InlineVariant& other) const {
  if (!isBinaryCompatible(terms, other.terms)) return false;

  return term_offset == other.term_offset && term_size == other.term_size &&
         tag_offset == other.tag_offset && tag_width == other.tag_width &&
         size == other.size && alignment == other.alignment;
}

bool OutlineVariant::isBinaryCompatibleWith(const OutlineVariant& other) const {
  if (!isBinaryCompatible(terms, other.terms)) return false;

  return tag_width == other.tag_width && tag_alignment == other.tag_alignment &&
         term_offset == other.term_offset &&
         term_alignment == other.term_alignment;
}

bool Array::isBinaryCompatibleWith(const Array& other) const {
  return elem.isBinaryCompatibleWith(other.elem) && length == other.length &&
         elem_size == other.elem_size && alignment == other.alignment;
}

bool Vector::isBinaryCompatibleWith(const Vector& other) const {
  return elem.isBinaryCompatibleWith(other.elem) &&
         min_length == other.min_length && max_length == other.max_length &&
         ppl_count == other.ppl_count && length_offset == other.length_offset &&
         length_size == other.length_size && ref_offset == other.ref_offset &&
         ref_size == other.ref_size && reference_mode == other.reference_mode &&
         inline_payload_offset == other.inline_payload_offset &&
         partial_payload_offset == other.partial_payload_offset;
}

bool Any::isBinaryCompatibleWith(const Any& other) const {
  // TODO: should this method ever return true? The only current application of
  // checking binary compatibility is for wire types, and Any never exists on
  // the wire.
  return data_ref_width == other.data_ref_width &&
         data_ref_offset == other.data_ref_offset &&
         type_ref_width == other.type_ref_width &&
         type_ref_offset == other.type_ref_offset && size == other.size &&
         alignment == other.alignment &&
         self.isBinaryCompatibleWith(other.self);
}

bool Protocol::isBinaryCompatibleWith(const Protocol& other) const {
  return head.isBinaryCompatibleWith(other.head) &&
         buffer_offset == other.buffer_offset;
}

}  // namespace types
}  // namespace pj
