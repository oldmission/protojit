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
struct ProtocolHead {};

}  // namespace gen

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

#include "pj/reflect.pj.hpp"

namespace pj {
namespace gen {

template <>
struct BuildPJType<Any> {
  static const void* build(PJContext* ctx) {
    return PJCreateAnyType(
        ctx, offsetof(Any, data_) << 3, sizeof(Any::data_) << 3,
        offsetof(Any, type_) << 3, sizeof(Any::type_) << 3, sizeof(Any) << 3,
        alignof(Any) << 3,
        ::pj::gen::BuildPJType<::pj::reflect::Protocol>::build(ctx));
  }
};

}  // namespace gen
}  // namespace pj

#endif  // PROTOJIT_PROTOJIT_HPP
