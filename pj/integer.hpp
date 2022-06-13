#include <cstdint>
#include <type_traits>

#include "runtime.h"
#include "traits.hpp"

namespace pj {

template <size_t width, PJSign S>
struct integer {};

}  // namespace pj

using pj_char = pj::integer<8, PJ_SIGN_SIGNLESS>;
using pj_int8 = pj::integer<8, PJ_SIGN_SIGNED>;
using pj_uint8 = pj::integer<8, PJ_SIGN_UNSIGNED>;

using pj_wchar = pj::integer<16, PJ_SIGN_SIGNLESS>;
using pj_int16 = pj::integer<16, PJ_SIGN_SIGNED>;
using pj_uint16 = pj::integer<16, PJ_SIGN_UNSIGNED>;

using pj_char32 = pj::integer<32, PJ_SIGN_SIGNLESS>;
using pj_int32 = pj::integer<32, PJ_SIGN_SIGNED>;
using pj_uint32 = pj::integer<32, PJ_SIGN_UNSIGNED>;

using pj_char64 = pj::integer<64, PJ_SIGN_SIGNLESS>;
using pj_int64 = pj::integer<64, PJ_SIGN_SIGNED>;
using pj_uint64 = pj::integer<64, PJ_SIGN_UNSIGNED>;

namespace pj {

template <>
struct wrapped_type<pj_char> {
  using type = char;
};

template <>
struct wrapped_type<pj_int8> {
  using type = int8_t;
};

template <>
struct wrapped_type<pj_uint8> {
  using type = uint8_t;
};

template <>
struct wrapped_type<pj_wchar> {
  using type = wchar_t;
};

template <>
struct wrapped_type<pj_int16> {
  using type = int16_t;
};

template <>
struct wrapped_type<pj_uint16> {
  using type = uint16_t;
};

template <>
struct wrapped_type<pj_char32> {
  using type = int32_t;
};

template <>
struct wrapped_type<pj_int32> {
  using type = int32_t;
};

template <>
struct wrapped_type<pj_uint32> {
  using type = uint32_t;
};

template <>
struct wrapped_type<pj_char64> {
  using type = int64_t;
};

template <>
struct wrapped_type<pj_int64> {
  using type = int64_t;
};

template <>
struct wrapped_type<pj_uint64> {
  using type = uint64_t;
};

namespace gen {
template <size_t width, PJSign sign>
struct BuildPJType<integer<width, sign>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    return PJCreateIntType(ctx, /*width=*/width, /*alignment=*/width,
                           /*sign=*/sign);
  }
};
}  // namespace gen

template <typename D, typename I>
struct integer_validate {
  using type = I;
  static_assert(std::is_same_v<D, typename pj::wrapped_type<I>::type>);
};

}  // namespace pj
