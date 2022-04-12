#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/arrays.pj.hpp"

namespace pj {

TEST_F(PJTest, ArraySameTest) {
  A1 x = {.arr = {'a', 'b', 'c', 'd'}};
  A1 y = {.arr = {0x7f, 0x7f, 0x7f, 0x7f}};

  transcode<A1>(&x, &y);

  for (intptr_t i = 0; i < 4; ++i) {
    EXPECT_EQ(y.arr[i], x.arr[i]);
  }
}

TEST_F(PJTest, ArrayExtendTest) {
  A1 x = {.arr = {'a', 'b', 'c', 'd'}};
  A2 y = {.arr = {0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f}};

  transcode<A1, A2>(&x, &y);

  for (intptr_t i = 0; i < 4; ++i) {
    EXPECT_EQ(y.arr[i], x.arr[i]);
  }
  EXPECT_EQ(y.arr[4], 0);
  EXPECT_EQ(y.arr[5], 0);
}

TEST_F(PJTest, ArrayDefaultTest) {
  A1 x = {.arr = {'a', 'b', 'c', 'd'}};
  A3 y = {.arr = {-1, -1, -1, -1},
          .arr2 = {A2{.arr = {-1, -1, -1, -1, -1, -1}},
                   A2{.arr = {-1, -1, -1, -1, -1, -1}},
                   A2{.arr = {-1, -1, -1, -1, -1, -1}}}};

  transcode<A1, A3>(&x, &y);

  for (intptr_t i = 0; i < 4; ++i) {
    EXPECT_EQ(y.arr[i], x.arr[i]);
  }
  for (intptr_t i = 0; i < 3; ++i) {
    for (intptr_t j = 0; j < 6; ++j) {
      EXPECT_EQ(y.arr2[i].arr[j], 0);
    }
  }
}

}  // namespace pj
