#include "protojit.hpp"
#include "util.hpp"

namespace pj {

struct AnyInt;
struct AnyUnit;
struct AnyStruct;
struct AnyVariant;
struct AnyArray;
struct AnyVector;

struct Any {
  Any() {}

  enum class Kind {
    Int,
    Unit,
    Struct,
    Variant,
    Array,
    Vector,
    Unknown,
  };

  Kind kind() const {
    switch (type().tag) {
      case reflect::Type::Kind::Int:
        return Kind::Int;
      case reflect::Type::Kind::Struct:
        return Kind::Struct;
      case reflect::Type::Kind::InlineVariant:
        throw std::logic_error("InlineVariant is not reflectable");
      case reflect::Type::Kind::OutlineVariant:
        return Kind::Variant;
      case reflect::Type::Kind::Unit:
        return Kind::Unit;
      case reflect::Type::Kind::Array:
        return Kind::Array;
      case reflect::Type::Kind::Vector:
        return Kind::Vector;
      case reflect::Type::Kind::undef:
        return Kind::Unknown;
    }
  }

 protected:
  Any(const reflect::Protocol* protocol, const char* data, int32_t offset)
      : protocol_(protocol), data_(data), offset_(offset) {}
  friend AnyStruct;

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

  size_t numFields() const { return type().value.Struct.fields.size(); }
  AnyField getField(size_t i) const;  // { return Field(*this, i); }

 private:
  friend AnyField;

  const reflect::Struct& strct() { return type().value.Struct; }

  Any getFieldData(size_t index) const {
    return Any(protocol_,
               data_ + type().value.Struct.fields[index].offset.bytes(),
               offset_ + type().value.Struct.fields[index].type);
  };
};

struct AnyField {
  AnyField operator*() { return *this; }

  std::string_view name() {
    auto name = owner_.strct().fields[index_].name;
    return {&name[0], name.size()};
  }

  Any value() { return owner_.getFieldData(index_); }

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

struct AnyInt : public Any {
  AnyInt(Any value) : Any(value) {
    ASSERT(type().tag == reflect::Type::Kind::Int);
  }

  Sign sign() const { return type().value.Int.sign; }
  Width width() const { return type().value.Int.width; }

  template <typename T>
  T getValue() {
    if (sign() == Sign::kSigned) {
      return getValueSigned<T>();
    }
    return getValueUnsigned<T>();
  }

 private:
#define CASE(TYPE, W)                       \
  case W: {                                 \
    TYPE##W##_t result;                     \
    memcpy(&result, data_, sizeof(result)); \
    return result;                          \
  }

  template <typename T>
  T getValueSigned() {
    switch (width().bits()) {
      CASE(int, 8)
      CASE(int, 16)
      CASE(int, 32)
      CASE(int, 64)
      default:
        throw std::logic_error("cannot interpret integer with bitwidth " +
                               std::to_string(width().bits()));
    }
  }

  template <typename T>
  T getValueUnsigned() {
    switch (width().bits()) {
      CASE(uint, 8)
      CASE(uint, 16)
      CASE(uint, 32)
      CASE(uint, 64)
      default:
        throw std::logic_error("cannot interpret integer with bitwidth " +
                               std::to_string(width().bits()));
    }
  }

#undef CASE
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
