#ifndef PROTOJIT_PROTOJIT_HPP
#define PROTOJIT_PROTOJIT_HPP

#include <array>
#include <cassert>
#include <memory>
#include <optional>

#include "arch_base.hpp"
#include "runtime.h"

namespace pj {

namespace gen {

template <typename T>
struct BuildPJType;

template <typename T>
struct BuildPJProtocol;

template <typename T>
struct ProtocolHead {};

}  // namespace gen

template <typename T, PJSign S>
struct Integer {
  static_assert(std::is_integral_v<T>);

 private:
  T pad_;
};

struct Unit {};

template <typename T, size_t MinLength, intptr_t MaxLength>
class ArrayView {
 public:
  ArrayView(const T* data, uint64_t length)
      : length(length), outline(reinterpret_cast<const char*>(data)) {
    assert(MaxLength < 0 || length <= static_cast<uint64_t>(MaxLength));
    if (length <= MinLength) {
      std::copy(data, data + length, storage.begin());
    }
  }

  template <size_t N, typename = std::enable_if_t<N <= MinLength>>
  ArrayView(const std::array<T, N>& arr) : length(N), outline(nullptr) {
    std::copy(arr.begin(), arr.end(), storage.begin());
  }

  ArrayView() : length(0) {}

  template <
      typename A = std::enable_if<!std::is_trivially_copy_assignable_v<T> &&
                                      std::is_copy_assignable_v<T>,
                                  ArrayView>>
  ArrayView& operator=(const ArrayView& o) {
    length = o.length;
    outline = o.outline;
    if (length <= MinLength) {
      for (intptr_t i = 0; i < length; ++i) {
        storage[i] = o.storage[i];
      }
    }
    return *this;
  }

  const T& operator[](uintptr_t i) const {
    if (length <= MinLength) {
      return storage[i];
    }
    return reinterpret_cast<const T*>(outline)[i];
  }

  template <typename U>
  bool operator==(const U& o) const {
    return std::equal(begin(), end(), o.begin(), o.end());
  }

  const T* begin() const {
    if (length <= MinLength) {
      return storage.begin();
    }
    return reinterpret_cast<const T*>(outline);
  }

  const T* end() const {
    if (length <= MinLength) {
      return storage.begin() + length;
    }
    return reinterpret_cast<const T*>(outline) + length;
  }

  uint64_t size() const { return length; }

  bool has_ref() const { return length > MinLength; }

 private:
  uint64_t length;
  const char* outline;
  std::array<T, MinLength> storage;

  template <typename U>
  friend struct gen::BuildPJType;
};

template <typename T, intptr_t MaxLength>
class ArrayView<T, 0, MaxLength> {
 public:
  ArrayView(const T* data, uint64_t length)
      : length(length), outline(reinterpret_cast<const char*>(data)) {
    assert(MaxLength < 0 || length <= static_cast<uint64_t>(MaxLength));
  }

  ArrayView() : length(0) {}

  ArrayView& operator=(const ArrayView& o) = default;

  const T& operator[](uintptr_t i) const {
    return reinterpret_cast<const T*>(outline)[i];
  }

  template <typename U>
  bool operator==(const U& o) const {
    return std::equal(begin(), end(), o.begin(), o.end());
  }

  const T* begin() const { return reinterpret_cast<const T*>(outline); }

  const T* end() const { return reinterpret_cast<const T*>(outline) + length; }

  uint64_t size() const { return length; }

 private:
  uint64_t length;
  const char* outline;

  template <typename U>
  friend struct gen::BuildPJType;
};

struct Any {
 private:
  const void* type_;
  const void* data_;

  template <typename U>
  friend struct gen::BuildPJType;
};

}  // namespace pj

namespace pj {
namespace gen {

template <typename T, PJSign S>
struct BuildPJType<Integer<T, S>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    static_assert(std::is_integral_v<T>);
    return PJCreateIntType(ctx, /*width=*/sizeof(T) << 3,
                           /*alignment=*/alignof(T) << 3,
                           /*sign=*/S);
  }
};

template <>
struct BuildPJType<::pj::Unit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    return PJCreateUnitType(ctx);
  }
};

template <typename Elem, size_t Length>
struct BuildPJType<std::array<Elem, Length>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    using Array = std::array<Elem, Length>;
    auto elem = BuildPJType<Elem>::build(ctx, domain);
    return PJCreateArrayType(ctx, /*elem=*/elem, /*length=*/Length,
                             /*elem_size=*/sizeof(Elem) << 3,
                             /*alignment=*/alignof(Array) << 3);
  }
};

template <typename Elem, size_t MinLength, intptr_t MaxLength>
struct BuildPJType<::pj::ArrayView<Elem, MinLength, MaxLength>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    using AV = ::pj::ArrayView<Elem, MinLength, MaxLength>;
    auto elem = BuildPJType<Elem>::build(ctx, domain);
    intptr_t inline_payload_offset = -1;
    intptr_t inline_payload_size = 0;
    if constexpr (MinLength > 0) {
      inline_payload_offset = offsetof(AV, storage) << 3;
      inline_payload_size = sizeof(AV::storage) << 3;
    }
    return PJCreateVectorType(ctx, /*elem=*/elem, /*min_length=*/MinLength,
                              /*max_length=*/MaxLength,
                              /*wire_min_length=*/MinLength,
                              /*ppl_count=*/0,
                              /*length_offset=*/offsetof(AV, length) << 3,
                              /*length_size=*/sizeof(AV::length) << 3,
                              /*ref_offset=*/offsetof(AV, outline) << 3,
                              /*ref_size=*/sizeof(AV::outline) << 3,
                              /*reference_mode=*/PJ_REFERENCE_MODE_POINTER,
                              inline_payload_offset, inline_payload_size,
                              /*partial_payload_offset=*/-1,
                              /*partial_payload_size=*/0,
                              /*size=*/sizeof(AV) << 3,
                              /*alignment=*/alignof(AV) << 3,
                              /*outlined_payload_alignment=*/64);
  }
};

}  // namespace gen
}  // namespace pj

#include "pj/reflect.pj.hpp"

namespace pj {
namespace gen {

template <>
struct BuildPJType<Any> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    return PJCreateAnyType(
        ctx, offsetof(Any, data_) << 3, sizeof(Any::data_) << 3,
        offsetof(Any, type_) << 3, sizeof(Any::type_) << 3, sizeof(Any) << 3,
        alignof(Any) << 3,
        ::pj::gen::BuildPJType<::pj::reflect::Protocol>::build(ctx, domain));
  }
};

}  // namespace gen
}  // namespace pj

#endif  // PROTOJIT_PROTOJIT_HPP
