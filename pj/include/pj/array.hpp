#pragma once

#include <array>

#include "runtime.h"
#include "traits.hpp"

namespace pj {

template <typename T, size_t N>
struct array : public std::array<typename wrapped_type<T>::type, N> {};

#if __cpp_deduction_guides >= 201606
template <typename _Tp, typename... _Up>
array(_Tp, _Up...)
    -> array<std::enable_if_t<(std::is_same_v<_Tp, _Up> && ...), _Tp>,
             1 + sizeof...(_Up)>;
#endif

namespace gen {

template <typename Elem, size_t Length>
struct BuildPJType<array<Elem, Length>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    using Array = array<Elem, Length>;
    auto elem = BuildPJType<Elem>::build(ctx, domain);
    return PJCreateArrayType(ctx, /*elem=*/elem, /*length=*/Length,
                             /*elem_size=*/sizeof(Elem) << 3,
                             /*alignment=*/alignof(Array) << 3);
  }
};

}  // namespace gen
}  // namespace pj
