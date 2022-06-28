#pragma once

#include <cstdint>

struct CoordinateA {
  int64_t x;
  int64_t y;
};

struct CoordinateB {
  int64_t x;
  long _;
  int64_t y;
};
