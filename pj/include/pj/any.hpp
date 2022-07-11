#pragma once

#include <pj/reflect.pj.hpp>
#include <pj/util.hpp>

namespace pj {

struct AnyInt;
struct AnyUnit;
struct AnyStruct;
struct AnyVariant;
struct AnySequence;

struct Any {
  Any() {}

  enum class Kind {
    Int,
    Unit,
    Struct,
    Variant,
    Sequence,
    Unknown,
  };

  Kind kind() const {
    if (!protocol_) return Kind::Unit;
    switch (type().tag) {
      case reflect::Type::Kind::Int:
        return Kind::Int;
      case reflect::Type::Kind::Struct:
        return Kind::Struct;
      case reflect::Type::Kind::InlineVariant:
        return Kind::Variant;
      case reflect::Type::Kind::OutlineVariant:
        throw std::logic_error("OutlineVariant is not reflectable");
      case reflect::Type::Kind::Unit:
        return Kind::Unit;
      case reflect::Type::Kind::Array:
      case reflect::Type::Kind::Vector:
        return Kind::Sequence;
      case reflect::Type::Kind::undef:
        return Kind::Unknown;
    }
  }

  bool operator==(const Any& o) const {
    return o.protocol_ == protocol_ && o.data_ == data_ && o.offset_ == offset_;
  }
  bool operator!=(const Any& o) const { return !(o == *this); }

 protected:
  Any(const reflect::Protocol* protocol, const char* data, int32_t offset)
      : protocol_(protocol), data_(data), offset_(offset) {}
  friend AnyStruct;
  friend AnyVariant;
  friend AnySequence;

  const reflect::Type& type() const { return protocol_->types.base()[offset_]; }

  const reflect::Protocol* protocol_ = nullptr;
  const char* data_ = nullptr;
  int32_t offset_ = 0;

  template <typename U>
  friend struct gen::BuildPJType;
};

struct AnyField;

struct AnyStruct : public Any {
  AnyStruct(Any value) : Any(value) {
    ASSERT(type().tag == reflect::Type::Kind::Struct);
  }

  AnyField begin() const;
  AnyField end() const;

  reflect::QualifiedName name() const { return strct().name; }

  size_t numFields() const { return type().value.Struct.fields.size(); }
  AnyField getField(size_t i) const;

 private:
  friend AnyField;

  const reflect::Struct& strct() const { return type().value.Struct; }

  Any getFieldData(size_t index) const {
    return Any(protocol_,
               data_ + type().value.Struct.fields[index].offset.bytes(),
               offset_ + type().value.Struct.fields[index].type);
  };
};

struct AnyField {
  AnyField operator*() { return *this; }
  AnyField* operator->() { return this; }
  AnyField& operator++() {
    ++index_;
    return *this;
  }

  bool operator==(const AnyField& o) const {
    return o.owner_ == owner_ && o.index_ == index_;
  }
  bool operator!=(const AnyField& o) const { return !(o == *this); }

  std::string_view name() const {
    auto name = owner_.strct().fields[index_].name;
    return {&name[0], name.size()};
  }

  Any value() const { return owner_.getFieldData(index_); }

 private:
  friend AnyStruct;
  AnyField(AnyStruct owner, size_t index) : owner_(owner), index_(index) {}

  AnyStruct owner_;
  size_t index_;
};

inline AnyField AnyStruct::begin() const { return AnyField(*this, 0); }
inline AnyField AnyStruct::end() const {
  return AnyField(*this, type().value.Struct.fields.size());
}
inline AnyField AnyStruct::getField(size_t i) const {
  return AnyField(*this, i);
}

#define CASE(TYPE, W)                       \
  case W: {                                 \
    TYPE##W##_t result;                     \
    memcpy(&result, data_, sizeof(result)); \
    return result;                          \
  }

template <typename T>
T getIntValueSigned(const void* data_, Width width) {
  switch (width.bits()) {
    CASE(int, 8)
    CASE(int, 16)
    CASE(int, 32)
    CASE(int, 64)
    default:
      throw std::logic_error("cannot interpret integer with bitwidth " +
                             std::to_string(width.bits()));
  }
}

template <typename T>
T getIntValueUnsigned(const void* data_, Width width) {
  switch (width.bits()) {
    CASE(uint, 8)
    CASE(uint, 16)
    CASE(uint, 32)
    CASE(uint, 64)
    default:
      throw std::logic_error("cannot interpret integer with bitwidth " +
                             std::to_string(width.bits()));
  }
}

#undef CASE

template <typename T>
T getIntValue(const void* data, Sign sign, Width width) {
  return sign == Sign::kSigned ? getIntValueSigned<T>(data, width)
                               : getIntValueUnsigned<T>(data, width);
}

struct AnyInt : public Any {
  AnyInt(Any value) : Any(value) {
    ASSERT(type().tag == reflect::Type::Kind::Int);
  }

  Sign sign() const { return type().value.Int.sign; }
  Width width() const { return type().value.Int.width; }

  template <typename T>
  T getValue() {
    return getIntValue<T>(data_, sign(), width());
  }
};

struct AnyVariant : public Any {
  AnyVariant(Any value) : Any(value) {
    ASSERT(type().tag == reflect::Type::Kind::InlineVariant);
  }

  bool isEnum() const {
    return std::all_of(variant().terms.begin(), variant().terms.end(),
                       [this](const auto& term) {
                         auto type =
                             protocol_->types.base()[offset_ + term.type];
                         return type.tag == reflect::Type::Kind::Unit;
                       });
  }

  uint64_t termTag() const {
    return getIntValueUnsigned<uint64_t>(data_ + variant().tag_offset.bytes(),
                                         variant().tag_width);
  }

  std::string_view termName() const {
    const uint64_t tag = termTag();
    for (auto& term : variant().terms) {
      if (term.tag == tag) {
        return {term.name.base(), term.name.size()};
      }
    }
    UNREACHABLE();
  }

  Any term() const {
    const uint64_t tag = termTag();
    for (auto& term : variant().terms) {
      if (term.tag == tag) {
        return Any(protocol_, data_ + variant().term_offset.bytes(),
                   offset_ + term.type);
      }
    }
    UNREACHABLE();
  }

 private:
  const reflect::InlineVariant& variant() const {
    return type().value.InlineVariant;
  }
};

struct AnySequence : public Any {
  AnySequence(Any value) : Any(value) {
    ASSERT(type().tag == reflect::Type::Kind::Vector ||
           type().tag == reflect::Type::Kind::Array);
  }

 private:
  struct iterator;

 public:
  size_t size() const {
    if (type().tag == reflect::Type::Kind::Array) {
      return type().value.Array.length;
    }
    return getIntValueUnsigned<size_t>(
        data_ + type().value.Vector.length_offset.bytes(),
        type().value.Vector.length_size);
  }

  iterator begin() const { return iterator(*this, 0); }
  iterator end() const { return iterator(*this, size()); }
  Any operator[](size_t i) const {
    if (type().tag == reflect::Type::Kind::Array) {
      return Any(protocol_, data_ + i * type().value.Array.elem_size.bytes(),
                 offset_ + type().value.Array.elem);
    }
    ASSERT(type().tag == reflect::Type::Kind::Vector);
    ASSERT(type().value.Vector.reference_mode == ReferenceMode::kPointer);
    ASSERT(type().value.Vector.min_length == 0);
    ASSERT(type().value.Vector.ppl_count == 0);
    const size_t pointer = getIntValueUnsigned<size_t>(
        data_ + type().value.Vector.ref_offset.bytes(),
        type().value.Vector.ref_size);
    return Any(protocol_,
               reinterpret_cast<const char*>(pointer) +
                   i * type().value.Vector.elem_width.bytes(),
               offset_ + type().value.Vector.elem);
  }

  bool isString() {
    int32_t elem_offset;
    if (type().tag == reflect::Type::Kind::Array) {
      elem_offset = type().value.Array.elem;
    } else {
      ASSERT(type().tag == reflect::Type::Kind::Vector);
      elem_offset = type().value.Vector.elem;
    }
    auto elem = protocol_->types.base()[offset_ + elem_offset];
    return elem.tag == reflect::Type::Kind::Int &&
           elem.value.Int.sign == Sign::kSignless &&
           elem.value.Int.width == Bytes(1);
  }

 private:
  struct iterator {
    Any operator*() const { return AnySequence(parent_)[index_]; }
    void operator++() { ++index_; }

    bool operator==(const iterator& o) const {
      return o.parent_ == parent_ && o.index_ == index_;
    }
    bool operator!=(const iterator& o) const { return !(o == *this); }

   private:
    iterator(Any parent, size_t index) : parent_(parent), index_(index) {}

    friend AnySequence;
    Any parent_;
    size_t index_;
  };
};

namespace gen {
template <>
struct BuildPJType<Any> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    return PJCreateAnyType(
        ctx, offsetof(Any, data_) << 3, sizeof(Any::data_) << 3,
        offsetof(Any, protocol_) << 3, sizeof(Any::protocol_) << 3,
        offsetof(Any, offset_) << 3, sizeof(Any::offset_) << 3,
        sizeof(Any) << 3, alignof(Any) << 3,
        ::pj::gen::BuildPJType<::pj::reflect::Protocol>::build(ctx, domain));
  }
};

}  // namespace gen
}  // namespace pj
