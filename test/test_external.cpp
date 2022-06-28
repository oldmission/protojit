#include <gtest/gtest.h>

#include "harness.hpp"

#include "test/external.pj.hpp"

namespace pj {

TEST_F(PJTest, ExternalTestA2B) {
  CoordinateA a{.x = 42, .y = 43};
  CoordinateB b{};

  transcode(Options<CoordinateA, CoordinateB>{.from = &a, .to = &b});

  EXPECT_EQ(b.x, 42);
  EXPECT_EQ(b.y, 43);
}

TEST_F(PJTest, ExternalTestB2A) {
  CoordinateB b{.x = 42, .y = 43};
  CoordinateA a{};

  transcode(Options<CoordinateB, CoordinateA>{.from = &b, .to = &a});

  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(a.y, 43);
}

}  // namespace pj
