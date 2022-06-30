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

TEST_F(PJTest, ExternalTestBoundingBox) {
  BoundingBox bb0{.tl = {.x = 3, .y = 4}, .br = {.x = 10, .y = 20}};
  BoundingBox bb1;

  transcode(Options<BoundingBox>{.from = &bb0, .to = &bb1});

  EXPECT_EQ(bb1.tl.x, 3);
  EXPECT_EQ(bb1.tl.y, 4);
  EXPECT_EQ(bb1.br.x, 10);
  EXPECT_EQ(bb1.br.y, 20);
}

}  // namespace pj
