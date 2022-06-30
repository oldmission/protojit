#ifndef PROTOJIT_PROTOJIT_HPP
#define PROTOJIT_PROTOJIT_HPP

#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <optional>

#include "arch_base.hpp"
#include "integer.hpp"
#include "runtime.h"
#include "span.hpp"
#include "traits.hpp"

namespace pj {

struct Unit {};

template <typename T, size_t N>
struct array : public std::array<typename wrapped_type<T>::type, N> {};

#if __cpp_deduction_guides >= 201606
template <typename _Tp, typename... _Up>
array(_Tp, _Up...)
    -> array<std::enable_if_t<(std::is_same_v<_Tp, _Up> && ...), _Tp>,
             1 + sizeof...(_Up)>;
#endif

struct Any {
 private:
  const void* type_;
  const void* data_;

  template <typename U>
  friend struct gen::BuildPJType;
};

template <typename T, typename S>
using DecodeHandler = void (*)(const T* msg, S* state);

template <typename T>
using SizeFunction = uintptr_t (*)(const T*);
template <typename T>
using EncodeFunction = void (*)(const T*, char*);
template <typename T, typename S>
using DecodeFunction = BoundedBuffer (*)(const char*, T*, BoundedBuffer,
                                         DecodeHandler<T, S>[], S*);

}  // namespace pj

namespace pj {
namespace gen {

template <>
struct BuildPJType<::pj::Unit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    return PJCreateUnitType(ctx);
  }
};

template <typename Elem, size_t Length>
struct BuildPJType<pj::array<Elem, Length>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    using Array = pj::array<Elem, Length>;
    auto elem = BuildPJType<Elem>::build(ctx, domain);
    return PJCreateArrayType(ctx, /*elem=*/elem, /*length=*/Length,
                             /*elem_size=*/sizeof(Elem) << 3,
                             /*alignment=*/alignof(Array) << 3);
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
