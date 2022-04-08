#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "test/structs.pj.hpp"

namespace pj {

TEST_F(PJTest, StructRemoveFieldSecondTest) {
  S_xy a{.x = 42, .y = 43};
  S_x b{.x = 0};

  transcode<S_xy, S_x>(&a, &b);

  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(a.y, 43);
  EXPECT_EQ(b.x, 42);
}

TEST_F(PJTest, StructRemoveFieldFirstTest) {
  S_xy a{.x = 42, .y = 43};
  S_y b{.y = 0};

  transcode<S_xy, S_y>(&a, &b);

  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(a.y, 43);
  EXPECT_EQ(b.y, 43);
}

// TODO: implement DefaultOp lowering
#if 0
TEST_F(PJTest, StructAddFieldSecondTest) {
  S_x a{.x = 42};
  S_xy b{.x = 0, .y = 0xff};

  transcode<S_x, S_xy>(&a, &b);

  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(b.x, 42);
  EXPECT_EQ(b.y, 0);
}

TEST_F(PJTest, StructAddFieldFirstTest) {
  S_y a{.y = 42};
  S_xy b{.x = 0xff, .y = 0xff};

  transcode<S_y, S_xy>(&a, &b);

  EXPECT_EQ(a.y, 42);
  EXPECT_EQ(b.x, 0);
  EXPECT_EQ(b.y, 42);
}
#endif

TEST_F(PJTest, NestedStructSameTest) {
  SO x{.s1 = {.x = 2, .y = 3}, .s2 = {.x = 4, .y = 5}};
  SO y = {};

  transcode<SO>(&x, &y);

  EXPECT_EQ(y.s1.x, 2);
  EXPECT_EQ(y.s1.y, 3);
  EXPECT_EQ(y.s2.x, 4);
  EXPECT_EQ(y.s2.y, 5);
}

TEST_F(PJTest, NestedStructRemoveInnerFieldSmallTest) {
  SO x{.s1 = {.x = 2, .y = 3}, .s2 = {.x = 4, .y = 5}};
  SOS y = {};

  transcode<SO, SOS>(&x, &y);

  EXPECT_EQ(y.s1.x, 2);
  EXPECT_EQ(y.s2.x, 4);
  EXPECT_EQ(y.s2.y, 5);
}

TEST_F(PJTest, NestedStructRemoveInnerFieldLargeTest) {
  TO x{.x = {.x = 2, .y = 3}, .y = {.x = 4, .y = 5}};
  TO2 y = {};

  transcode<TO, TO2>(&x, &y);

  EXPECT_EQ(y.x.x, 2);
  EXPECT_EQ(y.y.x, 4);
}

}  // namespace pj
