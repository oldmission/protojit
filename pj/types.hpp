#pragma once

#include <limits>
#include <numeric>
#include <sstream>
#include <variant>

#include <pj/util.hpp>

#include "arch.hpp"
#include "array_ref.hpp"
#include "type_support.hpp"

namespace pj {
namespace types {

using llvm::StringRef;

#define FOR_EACH_VALUE_TYPE(V)       \
  V(::pj::types::IntType)            \
  V(::pj::types::FloatType)          \
  V(::pj::types::UnitType)           \
  V(::pj::types::StructType)         \
  V(::pj::types::InlineVariantType)  \
  V(::pj::types::OutlineVariantType) \
  V(::pj::types::ArrayType)          \
  V(::pj::types::VectorType)         \
  V(::pj::types::AnyType)            \
  V(::pj::types::ProtocolType)

struct PrimitiveType : public ValueType {
  using ValueType::ValueType;

  static bool classof(mlir::Type val);

  mlir::Type toMLIR() const;
  mlir::Attribute getDefaultValue() const;
};

struct Int {
  /*** Parsed ***/
  Width width;

  /*** Generated ***/
  Width alignment;

  /*** Parsed ***/
  Sign sign;

  bool operator==(const Int& other) const {
    return width == other.width && sign == other.sign &&
           alignment == other.alignment;
  }

  bool isBinaryCompatibleWith(const Int& other) const { return *this == other; }

  Width headSize() const { return width; }
  Width headAlignment() const { return alignment; }
  bool hasMaxSize() const { return true; }
  ChildVector children() const { return {}; }
  bool hasDetails() const { return false; }
  void printDetails(llvm::raw_ostream& os) const {}
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Int& I) {
  switch (I.sign) {
    case Sign::kSigned:
      os << "i";
      break;
    case Sign::kUnsigned:
      os << "u";
      break;
    case Sign::kSignless:
      os << "c";
      break;
  }
  return os << I.width.bits() << "/" << I.alignment.bits();
}

inline ::llvm::hash_code hash_value(const Int& I) {
  using ::llvm::hash_value;
  return llvm::hash_combine(hash_value(I.width), hash_value(I.sign),
                            hash_value(I.alignment));
}

inline Int type_intern(__attribute__((unused))
                       mlir::TypeStorageAllocator& allocator,
                       const Int& I) {
  ASSERT(I.alignment.bits() != kNone);
  return I;
}

struct IntType : public mlir::Type::TypeBase<
                     IntType, StructuralTypeBase<Int, IntType, PrimitiveType>,
                     StructuralTypeStorage<Int, IntType>> {
  using Base::Base;
  using Base::get;
};

struct Float {
  /*** Parsed ***/
  enum FloatWidth { k32 = 32, k64 = 64 } width;

  /*** Generated ***/
  Width alignment;

  bool operator==(const Float& other) const {
    return width == other.width && alignment == other.alignment;
  }

  bool isBinaryCompatibleWith(const Float& other) const {
    return *this == other;
  }

  Width headSize() const { return Bits(width); }
  Width headAlignment() const { return alignment; }
  bool hasMaxSize() const { return true; }
  ChildVector children() const { return {}; }
  bool hasDetails() const { return false; }
  void printDetails(llvm::raw_ostream& os) const {}
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Float& F) {
  return os << "f" << F.width << "/" << F.alignment.bits();
}

inline ::llvm::hash_code hash_value(const Float& F) {
  using ::llvm::hash_value;
  return llvm::hash_combine(hash_value(F.width), hash_value(F.alignment));
}

inline Float type_intern(__attribute__((unused))
                         mlir::TypeStorageAllocator& allocator,
                         const Float& F) {
  ASSERT(F.alignment.bits() != kNone);
  return F;
}

struct FloatType
    : public mlir::Type::TypeBase<
          FloatType, StructuralTypeBase<Float, FloatType, PrimitiveType>,
          StructuralTypeStorage<Float, FloatType>> {
  using Base::Base;
  using Base::get;
};

inline bool PrimitiveType::classof(mlir::Type val) {
  return val.isa<IntType>() || val.isa<FloatType>();
}

inline mlir::Type PrimitiveType::toMLIR() const {
  if (auto i = dyn_cast<IntType>()) {
    return mlir::IntegerType::get(getContext(), i->width.bits(),
                                  mlir::IntegerType::Signless);
  } else if (auto f = dyn_cast<FloatType>()) {
    switch (f->width) {
      case Float::k32:
        return mlir::FloatType::getF32(getContext());
      case Float::k64:
        return mlir::FloatType::getF64(getContext());
    }
    UNREACHABLE();
  } else {
    UNREACHABLE();
  }
}

inline mlir::Attribute PrimitiveType::getDefaultValue() const {
  if (auto i = dyn_cast<IntType>()) {
    assert(i->width.IsNotNone());
    return mlir::IntegerAttr::get(toMLIR(), 0);
  } else if (auto f = dyn_cast<FloatType>()) {
    auto value = [&]() {
      switch (f->width) {
        case Float::k32:
          static_assert(sizeof(float) == 4);
          static_assert(std::numeric_limits<float>::has_quiet_NaN);
          return llvm::APFloat{std::numeric_limits<float>::quiet_NaN()};
        case Float::k64:
          static_assert(sizeof(double) == 8);
          static_assert(std::numeric_limits<double>::has_quiet_NaN);
          return llvm::APFloat{std::numeric_limits<double>::quiet_NaN()};
        default:
          UNREACHABLE();
      }
    }();
    return mlir::FloatAttr::get(toMLIR(), value);
  } else {
    UNREACHABLE();
  }
}

struct UnitTypeStorage : public ValueTypeStorage {
  using KeyTy = std::tuple<>;

  UnitTypeStorage() = default;

  bool operator==(const std::tuple<>) const { return true; }
  static llvm::hash_code hashKey(std::tuple<>) { return 0; }
  static UnitTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                    std::tuple<>) {
    return new (allocator.allocate<UnitTypeStorage>()) UnitTypeStorage();
  }
  void print(llvm::raw_ostream& os) const override { os << "<unit>"; }
  bool hasDetails() const override { return false; }
  void printDetails(llvm::raw_ostream& os) const override {}
  Width headSize() const override { return Bytes(0); }
  Width headAlignment() const override { return Bytes(1); }
  bool hasMaxSize() const override { return true; }
  ChildVector children() const override { return {}; }
  bool isBinaryCompatibleWith(ValueType type) const override { return true; }
};

struct UnitType
    : public mlir::Type::TypeBase<UnitType, ValueType, UnitTypeStorage> {
  using Base::Base;
  using Base::get;
};

struct StructField {
  /*** Parsed ***/
  ValueType type;
  StringRef name;

  /*** Generated ***/
  Width offset;

  bool operator==(const StructField& other) const {
    return type == other.type && name == other.name && offset == other.offset;
  }

  bool operator<(const StructField& other) const { return name < other.name; }
};

struct Struct {
  /*** Parsed ***/
  // Invariant: fields are listed in alphabetical order.
  ArrayRef<StructField> fields = {};

  /*** Generated ***/
  Width size = Width::None();
  Width alignment = Width::None();

  // Set during internment.
  bool has_max_size = false;
  ValueType outline_variant = {};

  bool isBinaryCompatibleWith(const Struct& other) const;

  Width headSize() const { return size; }
  Width headAlignment() const { return alignment; }
  bool hasMaxSize() const { return has_max_size; }
  ChildVector children() const {
    ChildVector children;
    for (const StructField& f : fields) {
      children.push_back(std::make_pair(
          f.type, f.name.str() + " @ " + std::to_string(f.offset.bits())));
    }
    return children;
  }
  bool hasDetails() const { return true; }
  void printDetails(llvm::raw_ostream& os) const {
    os << "size: " << size.bits() << ", alignment: " << alignment.bits();
  }
};

Struct type_intern(mlir::TypeStorageAllocator& allocator, const Struct& key);

struct StructType
    : public mlir::Type::TypeBase<
          StructType, NominalTypeBase<NominalType, Struct, StructType>,
          NominalTypeStorage<Struct, StructType>> {
  using Base::Base;
  using Base::get;

  friend struct NominalTypeBase<NominalType, Struct, StructType>;
};

struct TermAttribute {
  // All attribute values must have field is_default. When an unknown attribute
  // is decoded from a counterparty, it gets decoded as a variant with term
  // kUndef and value Undef.

  struct Undef {
    bool is_default;  // Whether this term should be used as the default.

    bool operator==(const Undef& other) const {
      return is_default == other.is_default;
    }
  };

  struct VectorSplit {
    enum Type { kInline, kOutline } type;
    uint64_t inline_length;
    PathAttr path;  // Relative path from term type to the vector.

    bool is_default() const { return type == kOutline; }

    bool operator==(const VectorSplit& other) const {
      return type == other.type && inline_length == other.inline_length &&
             path == other.path;
    }
  };

  std::variant<Undef, VectorSplit> value;

  bool operator==(const TermAttribute& other) const {
    return value == other.value;
  }

  std::string toString() const;
};

struct Term {
  /*** Parsed ***/
  StringRef name;
  ValueType type;
  uint64_t tag;  // Must be >0. 0 is reserved for UNDEF.

  /*** Generated ***/
  ArrayRef<TermAttribute> attributes;

  bool operator==(const Term& other) const {
    return name == other.name && type == other.type && tag == other.tag &&
           attributes == other.attributes;
  }

  bool operator<(const Term& other) const { return name < other.name; }

  std::string toString() const;
};

// Variants can have two representations.
//
// In the inline representation, the term and tag are stored (roughly) adjacent
// in memory, within a region of static size. This entails that the static size
// is at least as large as the size of the largest term.
struct InlineVariant {
  /*** Parsed ***/
  ArrayRef<Term> terms = {};
  uint64_t default_term;

  /*** Generated ***/
  // Invariant: term and tag should not overlap.
  Width term_offset = Width::None();
  Width term_size = Width::None();
  Width tag_offset = Width::None();
  Width tag_width = Width::None();

  Width size = Width::None();
  Width alignment = Width::None();

  bool isBinaryCompatibleWith(const InlineVariant& other) const;

  Width headSize() const { return size; }
  Width headAlignment() const { return alignment; }
  bool hasMaxSize() const {
    // TODO: implement logic to determine if it can have a max size and update
    // SizeOp lowering accordingly.
    return false;
  }
  ChildVector children() const {
    ChildVector children;
    for (const Term& t : terms) {
      children.push_back(std::make_pair(t.type, t.toString()));
    }
    return children;
  }
  bool hasDetails() const { return true; }
  void printDetails(llvm::raw_ostream& os) const {
    os << "term: <term>[" << term_size.bits() << "] @ " << term_offset.bits();
    os << ", tag: u" << tag_width.bits() << " @ " << tag_offset.bits();
    os << ", size: " << size.bits() << ", alignment: " << alignment.bits();
  }

  // Set during internment.
  bool is_enum = false;
};

// Variants can have two representations.
//
// In the outline representation, the "head" of the variant only stores
// the tag. The body is found at a fixed offset from the head.
//
// This representation may only be used on the wire.
struct OutlineVariant {
  ArrayRef<Term> terms = {};
  uint64_t default_term;

  Width tag_width = Width::None();
  Width tag_alignment = Width::None();
  Width term_offset = Width::None();
  Width term_alignment = Width::None();

  bool isBinaryCompatibleWith(const OutlineVariant& other) const;

  Width headSize() const { return tag_width; }

  Width headAlignment() const {
    // Since the term is located at a fixed offset from the start of the
    // message, the message itself must be aligned to at least the alignment
    // demanded by the term.
    return std::max(tag_alignment, term_alignment);
  }

  bool hasMaxSize() const {
    // TODO: implement logic to determine if it can have a max size and update
    // SizeOp lowering accordingly.
    return false;
  }
  ChildVector children() const {
    ChildVector children;
    for (const Term& t : terms) {
      children.push_back(std::make_pair(t.type, t.toString()));
    }
    return children;
  }
  bool hasDetails() const { return true; }
  void printDetails(llvm::raw_ostream& os) const {
    os << "term: <term>/" << term_alignment.bits() << " @ "
       << term_offset.bits();
    os << ", tag: u" << tag_width.bits() << "/" << tag_alignment.bits();
  }
};

template <typename V>
inline V internVariant(mlir::TypeStorageAllocator& allocator,
                       const V& type_data) {
  // Must at least contain a default term.
  assert(type_data.terms.size() > 0);

  V result = type_data;

  // Sort the indices of the terms in alphabetical order of the terms.
  std::vector<size_t> indices(type_data.terms.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t l, size_t r) {
    return type_data.terms[l].name < type_data.terms[r].name;
  });

  result.default_term = std::distance(
      indices.begin(),
      std::find(indices.begin(), indices.end(), type_data.default_term));

  // Intern the terms in order of indices.
  auto* terms = reinterpret_cast<Term*>(
      allocator.allocate(sizeof(Term) * type_data.terms.size(), alignof(Term)));

  bool is_enum = true;
  for (uintptr_t i = 0; i < type_data.terms.size(); ++i) {
    const auto& term = type_data.terms[indices[i]];
    terms[i] = Term{
        .name = allocator.copyInto(term.name),
        .type = term.type,
        .tag = term.tag,
        .attributes =
            ArrayRef<TermAttribute>{allocator.copyInto(term.attributes)},
    };
    if (is_enum && !term.type.isUnit()) {
      is_enum = false;
    }
  }

  result.terms = {terms, type_data.terms.size()};
  if constexpr (std::is_same_v<V, InlineVariant>) {
    result.is_enum = is_enum;
  }
  return result;
}

inline InlineVariant type_intern(mlir::TypeStorageAllocator& allocator,
                                 const InlineVariant& type_data) {
  return internVariant(allocator, type_data);
}

inline OutlineVariant type_intern(mlir::TypeStorageAllocator& allocator,
                                  const OutlineVariant& type_data) {
  return internVariant(allocator, type_data);
}

struct VariantType : public NominalType {
  using NominalType::NominalType;

  static bool classof(mlir::Type val);

  ArrayRef<Term> terms() const;
  size_t default_term() const;
  Width tag_width() const;
  Width tag_offset() const;
  Width term_offset() const;
};

inline Width compute_tag_width(VariantType v) {
  Width w = pj::Bytes(1);
  if (v.terms().size() > 0) {
    auto max_tag = std::max_element(
        v.terms().begin(), v.terms().end(),
        [](const Term& a, const Term& b) { return a.tag < b.tag; });
    w = Bytes(RoundUpToPowerOfTwo(std::ceil(std::log2(max_tag->tag + 1) / 8)));
  }
  return w;
}

struct InlineVariantType
    : public mlir::Type::TypeBase<
          InlineVariantType,
          NominalTypeBase<VariantType, InlineVariant, InlineVariantType>,
          NominalTypeStorage<InlineVariant, InlineVariantType>> {
  using Base::Base;
  using Base::classof;
  using Base::get;

  ArrayRef<Term> terms() const { return (*this)->terms; }

  friend struct NominalTypeBase<VariantType, InlineVariant, InlineVariantType>;
};

struct OutlineVariantType
    : public mlir::Type::TypeBase<
          OutlineVariantType,
          NominalTypeBase<VariantType, OutlineVariant, OutlineVariantType>,
          NominalTypeStorage<OutlineVariant, OutlineVariantType>> {
  using Base::Base;
  using Base::classof;
  using Base::get;

  ArrayRef<Term> terms() const { return (*this)->terms; }

  friend struct NominalTypeBase<VariantType, OutlineVariant,
                                OutlineVariantType>;
};

inline ArrayRef<Term> VariantType::terms() const {
  if (auto v = dyn_cast<OutlineVariantType>()) {
    return v->terms;
  } else if (auto v = dyn_cast<InlineVariantType>()) {
    return v->terms;
  } else {
    UNREACHABLE();
  }
}

inline size_t VariantType::default_term() const {
  if (auto v = dyn_cast<OutlineVariantType>()) {
    return v->default_term;
  } else if (auto v = dyn_cast<InlineVariantType>()) {
    return v->default_term;
  } else {
    UNREACHABLE();
  }
}

inline Width VariantType::tag_width() const {
  if (auto v = dyn_cast<OutlineVariantType>()) {
    return v->tag_width;
  } else if (auto v = dyn_cast<InlineVariantType>()) {
    return v->tag_width;
  } else {
    UNREACHABLE();
  }
}

inline Width VariantType::term_offset() const {
  if (auto v = dyn_cast<OutlineVariantType>()) {
    return v->term_offset;
  }

  if (auto v = dyn_cast<InlineVariantType>()) {
    return v->term_offset;
  }

  UNREACHABLE();
}

inline Width VariantType::tag_offset() const {
  if (auto v = dyn_cast<OutlineVariantType>()) {
    return Bits(0);
  }

  if (auto v = dyn_cast<InlineVariantType>()) {
    return v->tag_offset;
  }

  UNREACHABLE();
}

inline bool VariantType::classof(mlir::Type val) {
  return val.isa<InlineVariantType>() || val.isa<OutlineVariantType>();
}

// Arrays are fixed-length sequences of values.
struct Array {
  /*** Parsed ***/
  ValueType elem;
  uint64_t length;

  /*** Generated ***/
  // elem_size may be larger than the size of the inner element type
  // in the event that extra padding is needed for alignment reasons.
  Width elem_size;

  // Alignment of the array itself, which must be at least as large
  // as the element type's alignment
  Width alignment;

  // Set during internment.
  bool has_max_size = false;

  bool operator==(const Array& ary) const {
    return elem == ary.elem && length == ary.length &&
           elem_size == ary.elem_size && alignment == ary.alignment;
  }

  bool isBinaryCompatibleWith(const Array& other) const;

  Width headSize() const { return elem_size * length; }
  Width headAlignment() const { return alignment; }
  bool hasMaxSize() const { return has_max_size; }
  ChildVector children() const {
    ChildVector children;
    children.push_back(std::make_pair(elem, "elem"));
    return children;
  }
  bool hasDetails() const { return false; }
  void printDetails(llvm::raw_ostream& os) const {}
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Array& A) {
  os << A.elem << "[" << A.length;

  if (A.elem_size != A.elem.headSize()) {
    os << "|" << A.elem_size;
  }

  os << "]";

  if (A.alignment != A.elem.headAlignment()) {
    os << "/" << A.alignment;
  }

  return os;
}

inline ::llvm::hash_code hash_value(const Array& ary) {
  using ::llvm::hash_value;
  return llvm::hash_combine(hash_value(ary.elem), hash_value(ary.length),
                            hash_value(ary.elem_size),
                            hash_value(ary.alignment));
}

inline Array type_intern(mlir::TypeStorageAllocator& alloc, Array ary) {
  ary.has_max_size = ary.elem.hasMaxSize();
  return ary;
}

struct ArrayType
    : public mlir::Type::TypeBase<ArrayType,
                                  StructuralTypeBase<Array, ArrayType>,
                                  StructuralTypeStorage<Array, ArrayType>> {
  using Base::Base;
  using Base::get;
};

// Vectors are variable-length sequences of values.
//
// The number of elements may be bounded above or below.
// When bounded below, it is a soft bound, acting as a hint that many/most
// instances of the vector will have at least that many elements. This hint
// is mainly used for allocating inline storage.
//
// When bounded above, it is an (inclusive) hard bound, meaning instances of the
// vector can never have more elements than the bound. This can be used for
// loop unrolling or minimizing bits used to represent the variable size.
//
// A vector can take on two different representations at runtime.
// For a given length N, either the payload is stored inline, if N <=
// min_length:
//
// Representation A
// <- - size - - - >
// +---------------+
// | N | <N elems> |
// +---------------+
//
// Or the payload is stored separately, if N > min_length:
//
//           Representation B
// <- - - - - - -size - - - - - - - ->
// +---------------------------------+
// | N |  <ref>  | <ppl_count elems> |
// +---------------------------------+
//
// Note that the fields may be in a different order than shown above, or have
// padding in between.
//
// Ref is either a pointer to the beginning of the outlined elements (of
// which there would be N - ppl_count many), or an offset from the beginning of
// the Vector body to the start of the outlined elements, depending on
// offset_mode. Only the offset mode can be used in protocol types, but both
// modes can be used in memory types.
//
// If min_length == 0, only representation B is possible.
// If min_length == max_length, only representation A is possible.
struct Vector {
  /*** Parsed ***/
  ValueType elem;

  // Never None. 0 implies no inline storage.
  uint64_t min_length;

  // May be None. May not be 0.
  int64_t max_length;
  int64_t maxLengthBound() const {
    return max_length >= 0 ? max_length : std::numeric_limits<intptr_t>::max();
  }

  /*** Generated ***/
  // The min_length specified by the user for the wire type.
  uint64_t wire_min_length;

  // Partial payload length -- number of elements stored inline, when the total
  // number of elements exceeds min_length. Never None, also never guaranteed to
  // be greater than 0.
  intptr_t ppl_count;

  Width length_offset;
  Width length_size;

  // Only usable for (B).
  // May be kNone iff min_length == max_length.
  Width ref_offset;
  Width ref_size;

  ReferenceMode reference_mode;

  // Both may be None iff min_length == 0.
  Width inline_payload_offset;
  Width inline_payload_size;

  // Both may be None even if min_length > 0, in the event
  // no inline elements fit along with the reference.
  //
  // Will always be None if min_length == max_length.
  Width partial_payload_offset;
  Width partial_payload_size;

  Width size;

  bool isBinaryCompatibleWith(const Vector& other) const;

  Width headSize() const { return size; }

  // Alignment must be at least as large as the element type's alignment
  // if min_length > 0.
  Width alignment;
  Width headAlignment() const { return alignment; }

  // At least as large as element type's alignment.
  // May be None if min_length == max_length.
  Width outlined_payload_alignment;

  // Set during internment.
  Width elem_width;
  bool has_max_size = false;
  bool hasMaxSize() const { return has_max_size; }
  ChildVector children() const {
    ChildVector children;
    children.push_back(std::make_pair(elem, "elem"));
    return children;
  }
  bool hasDetails() const { return true; }
  void printDetails(llvm::raw_ostream& os) const {
    os << "length: u" << length_size.bits() << " @ " << length_offset.bits();
    if (max_length < 0 ||
        (max_length >= 0 && static_cast<uint64_t>(max_length) > min_length)) {
      os << ", ref: u" << ref_size.bits() << " @ " << ref_offset.bits();
      if (ppl_count > 0) {
        os << ", ppl: " << elem << "[" << ppl_count;
        if (elem_width != elem.headSize()) {
          os << "|" << elem_width.bits();
        }
        os << "] @ " << partial_payload_offset.bits();
      }
    }
    if (min_length > 0) {
      os << ", inline: " << elem << "[" << min_length;
      if (elem_width != elem.headSize()) {
        os << "|" << elem_width.bits();
      }
      os << "] @ " << inline_payload_offset.bits();
    }
  }

  bool operator==(const Vector& V) const {
    return elem == V.elem && elem_width == V.elem_width &&
           min_length == V.min_length && wire_min_length == V.wire_min_length &&
           max_length == V.max_length && ppl_count == V.ppl_count &&
           length_offset == V.length_offset && length_size == V.length_size &&
           ref_offset == V.ref_offset && ref_size == V.ref_size &&
           reference_mode == V.reference_mode &&
           inline_payload_offset == V.inline_payload_offset &&
           inline_payload_size == V.inline_payload_size &&
           partial_payload_offset == V.partial_payload_offset &&
           partial_payload_size == V.partial_payload_size && size == V.size &&
           alignment == V.alignment &&
           outlined_payload_alignment == V.outlined_payload_alignment;
  }
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Vector& V) {
  os << V.elem << "[" << V.min_length << "(" << V.wire_min_length
     << "):" << V.max_length;

  if (V.elem_width != V.elem.headSize()) {
    os << "|" << V.elem_width;
  }

  os << "]";

  if (V.alignment != V.elem.headAlignment()) {
    os << "/" << V.alignment;
  }

  return os;
}

inline ::llvm::hash_code hash_value(const Vector& V) {
  using ::llvm::hash_value;
  return llvm::hash_combine(
      hash_value(V.elem), hash_value(V.elem_width), hash_value(V.min_length),
      hash_value(V.wire_min_length), hash_value(V.max_length),
      hash_value(V.ppl_count), hash_value(V.length_offset),
      hash_value(V.length_size), hash_value(V.ref_offset),
      hash_value(V.ref_size), hash_value(V.reference_mode),
      hash_value(V.inline_payload_offset), hash_value(V.inline_payload_size),
      hash_value(V.partial_payload_offset), hash_value(V.partial_payload_size),
      hash_value(V.size), hash_value(V.alignment),
      hash_value(V.outlined_payload_alignment));
}

inline Vector type_intern(mlir::TypeStorageAllocator& allocator, Vector V) {
  ASSERT(V.elem_width.IsNotNone());
  V.has_max_size = (V.max_length >= 0) ? V.elem.hasMaxSize() : false;
  return V;
}

struct VectorType
    : public mlir::Type::TypeBase<VectorType,
                                  StructuralTypeBase<Vector, VectorType>,
                                  StructuralTypeStorage<Vector, VectorType>> {
  using Base::Base;
  using Base::get;
};

// Any is a type which cannot be encoded, only decoded.
//
// It is so named because any type can be decoded into it without loss of
// information. It provides reflective capabilities, for example, to read a
// value of an unknown type from a counterparty and traverse the value in its
// original form.
//
// An instance of AnyType can only exist in memory, so data_ref_width and
// type_ref_width must be equal to the host word size.
struct Any {
  Width data_ref_width;
  Width data_ref_offset;

  Width protocol_ref_width;
  Width protocol_ref_offset;

  Width offset_width;
  Width offset_offset;

  Width size;
  Width alignment;

  ValueType self;

  bool operator==(const Any& other) const {
    return data_ref_width == other.data_ref_width &&
           data_ref_offset == other.data_ref_offset &&
           protocol_ref_width == other.protocol_ref_width &&
           protocol_ref_offset == other.protocol_ref_offset &&
           offset_width == other.offset_width &&
           offset_offset == other.offset_offset && size == other.size &&
           alignment == other.alignment && self == other.self;
  }

  bool isBinaryCompatibleWith(const Any& other) const;

  Width headSize() const { return size; }
  Width headAlignment() const { return alignment; }
  bool hasMaxSize() const { return false; }
  ChildVector children() const { return {}; }
  bool hasDetails() const { return true; }
  void printDetails(llvm::raw_ostream& os) const {}
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Any& A) {
  return os << "Any|...";
}

inline ::llvm::hash_code hash_value(const Any& A) {
  using ::llvm::hash_value;
  return llvm::hash_combine(
      hash_value(A.data_ref_width), hash_value(A.data_ref_offset),
      hash_value(A.protocol_ref_width), hash_value(A.protocol_ref_offset),
      hash_value(A.offset_width), hash_value(A.offset_offset),
      hash_value(A.size), hash_value(A.alignment), hash_value(A.self));
}

inline Any type_intern(mlir::TypeStorageAllocator& allocator, const Any& A) {
  return A;
}

struct AnyType
    : public mlir::Type::TypeBase<AnyType, StructuralTypeBase<Any, AnyType>,
                                  StructuralTypeStorage<Any, AnyType>> {
  using Base::Base;
  using Base::get;
};

struct Protocol {
  /*** Generated ***/
  ValueType head;
  Width buffer_offset = Bytes(0);

  bool operator==(const Protocol& P) const {
    return head == P.head && buffer_offset == P.buffer_offset;
  }
  bool isBinaryCompatibleWith(const Protocol& other) const;
  Width headSize() const { return head.headSize(); }
  Width headAlignment() const { return head.headAlignment(); }
  bool hasMaxSize() const { return head.hasMaxSize(); }
  ChildVector children() const { return head.children(); }
  bool hasDetails() const { return head.hasDetails(); }
  void printDetails(llvm::raw_ostream& os) const { head.printDetails(os); }
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const Protocol& proto) {
  os << "Proto(";
  proto.head.cast<ValueType>().print(os);
  return os << ")";
}

inline Protocol type_intern(mlir::TypeStorageAllocator& allocator,
                            const Protocol& P) {
  return P;
}

inline llvm::hash_code hash_value(const Protocol& P) {
  using ::llvm::hash_value;
  return hash_value(P.head);
}

struct ProtocolType
    : public mlir::Type::TypeBase<
          ProtocolType, StructuralTypeBase<Protocol, ProtocolType>,
          StructuralTypeStorage<Protocol, ProtocolType>> {
  using Base::Base;
  using Base::get;
};

// A value of Buffer type represents a pointer to some mutable byte buffer.
// This is used to represent external memory during serialization and
// deserialization.
//
// A BoundedBuffer carries a size so accesses may be bounds-checked.

// A fake Buffer type for size functions.
struct DummyBufferType
    : public mlir::Type::TypeBase<DummyBufferType, mlir::Type,
                                  mlir::TypeStorage> {
  using Base::Base;
  using Base::get;

  void print(llvm::raw_ostream& os) const { os << "dbuf"; }
};

struct RawBufferType : public mlir::Type::TypeBase<RawBufferType, mlir::Type,
                                                   mlir::TypeStorage> {
  using Base::Base;
  using Base::get;

  void print(llvm::raw_ostream& os) const { os << "rbuf"; }
};

struct BoundedBufferType
    : public mlir::Type::TypeBase<BoundedBufferType, mlir::Type,
                                  mlir::TypeStorage> {
  using Base::Base;
  using Base::get;

  void print(llvm::raw_ostream& os) const { os << "bbuf"; }
};

struct DispatchHandlerStorage : public mlir::AttributeStorage {
  using KeyTy = std::pair<PathAttr, uint64_t>;

  DispatchHandlerStorage(KeyTy key) : key(key) {}

  bool operator==(const KeyTy& k) const { return k == key; }

  static llvm::hash_code hashKey(const KeyTy& k) {
    using ::llvm::hash_value;
    return llvm::hash_value(k);
  }

  static DispatchHandlerStorage* construct(
      mlir::AttributeStorageAllocator& allocator, KeyTy key) {
    return new (allocator.allocate<PathAttrStorage>())
        DispatchHandlerStorage(key);
  }

  KeyTy key;
};

struct DispatchHandlerAttr
    : public mlir::Attribute::AttrBase<DispatchHandlerAttr, mlir::Attribute,
                                       DispatchHandlerStorage> {
  using Base::Base;
  using Base::get;

  DispatchHandlerAttr getValue() const { return *this; }

  void print(llvm::raw_ostream& os) const;
  PathAttr path() const { return getImpl()->key.first; }
  uint64_t index() const { return getImpl()->key.second; }
};

void printType(llvm::raw_ostream& os, mlir::Type type);

struct WidthAttributeStorage : public mlir::AttributeStorage {
  using KeyTy = Width;

  WidthAttributeStorage(Width value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy& key) const { return key == value; }

  static llvm::hash_code hashKey(const KeyTy& key) { return key.bits(); }

  /// Construct a new storage instance.
  static WidthAttributeStorage* construct(
      mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<WidthAttributeStorage>())
        WidthAttributeStorage(key);
  }

  Width value;
};

struct WidthAttr : public mlir::Attribute::AttrBase<WidthAttr, mlir::Attribute,
                                                    WidthAttributeStorage> {
  using Base::Base;
  using Base::get;

  void print(llvm::raw_ostream& os) { os << "b" << getValue().bits(); }
  Width* operator->() const { return &getImpl()->value; }
  Width getValue() const { return getImpl()->value; }
};

struct UserStateType : public mlir::Type::TypeBase<UserStateType, mlir::Type,
                                                   mlir::TypeStorage> {
  using Base::Base;
  mlir::Type toLLVM() const;

  static UserStateType get(mlir::MLIRContext* ctx) { return Base::get(ctx); }
};

struct HandlersArrayType
    : public mlir::Type::TypeBase<HandlersArrayType, mlir::Type,
                                  mlir::TypeStorage> {
  using Base::Base;
  mlir::Type toLLVM() const;

  static HandlersArrayType get(mlir::MLIRContext* ctx) {
    return Base::get(ctx);
  }
};

}  // namespace types
}  // namespace pj
