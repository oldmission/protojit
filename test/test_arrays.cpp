#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/arrays.pj.hpp"

namespace pj {

TEST_F(PJTest, ArraySameTest) {
  A1 x = {'a', 'b', 'c', 'd'};
  A1 y = {0x7f, 0x7f, 0x7f, 0x7f};

  Transcode<A1>(&x, &y);

  for (intptr_t i = 0; i < 4; ++i) {
    EXPECT_EQ(y[i], x[i]);
  }
}

TEST_F(PJTest, ArrayExtendTest) {
  A1 x = {'a', 'b', 'c', 'd'};
  A2 y = {0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f};

  Transcode<A1, A2>(&x, &y);

  for (intptr_t i = 0; i < 4; ++i) {
    EXPECT_EQ(y[i], x[i]);
  }
  EXPECT_EQ(y[4], 0);
  EXPECT_EQ(y[5], 0);
}

}  // namespace pj
