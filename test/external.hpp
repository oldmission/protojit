#pragma once

#include <cstdint>

#include "pj/integer.hpp"
#include "pj/runtime.h"
#include "pj/traits.hpp"

struct CoordinateA {
  int64_t x;
  int64_t y;
};

struct CoordinateB {
  int64_t x;
  long _;
  int64_t y;
};

struct Point {
  int32_t x;
  int32_t y;
};

namespace pj {
namespace gen {
template <>
struct BuildPJType<Point> {
  static const PJStructType* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[2];
    const auto* int_type = BuildPJType<pj_int32>::build(ctx, domain);
    fields[0] = PJCreateStructField(/*name=*/"x", /*type=*/int_type,
                                    /*offset=*/offsetof(Point, x) << 3);
    fields[1] = PJCreateStructField(/*name=*/"y", /*type=*/int_type,
                                    /*offset=*/offsetof(Point, y) << 3);
    const char* typname[1] = {"Point"};
    return PJCreateStructType(ctx, /*name_size=*/1, /*name=*/typname,
                              /*type_domain=*/domain,
                              /*num_fields=*/2, /*fields=*/fields,
                              /*size=*/sizeof(Point) << 3,
                              /*alignment=*/alignof(Point) << 3);
  }
};
}  // namespace gen
}  // namespace pj
