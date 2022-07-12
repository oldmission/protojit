#pragma once

#include "runtime.h"
#include "traits.hpp"

namespace pj {

namespace gen {

template <>
struct BuildPJType<float> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    static_assert(sizeof(float) == 4);
    return PJCreateFloatType(ctx, /*width=*/PJ_FLOAT_WIDTH_32,
                             /*alignment=*/alignof(float) << 3);
  }
};

template <>
struct BuildPJType<double> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    static_assert(sizeof(double) == 8);
    return PJCreateFloatType(ctx, /*width=*/PJ_FLOAT_WIDTH_64,
                             /*alignment=*/alignof(float) << 3);
  }
};

}  // namespace gen
}  // namespace pj
