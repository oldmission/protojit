#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "test/structs.pj.hpp"

namespace pj {

TEST_F(PJTest, StructRemoveFieldSecondTest) {
  S_xy a{.x = 42, .y = 43};
  S_x b{.x = 0};

  auto [_, enc_size] = transcode<S_xy, S_x>(&a, &b);

  EXPECT_EQ(enc_size, 12);
  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(a.y, 43);
  EXPECT_EQ(b.x, 42);
}

TEST_F(PJTest, StructRemoveFieldFirstTest) {
  S_xy a{.x = 42, .y = 43};
  S_y b{.y = 0};

  auto [_, enc_size] = transcode<S_xy, S_y>(&a, &b);

  EXPECT_EQ(enc_size, 12);
  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(a.y, 43);
  EXPECT_EQ(b.y, 43);
}

TEST_F(PJTest, StructAddFieldSecondTest) {
  S_x a{.x = 42};
  S_xy b{.x = 0, .y = 0xff};

  auto [_, enc_size] = transcode<S_x, S_xy>(&a, &b);

  EXPECT_EQ(enc_size, 8);
  EXPECT_EQ(a.x, 42);
  EXPECT_EQ(b.x, 42);
  EXPECT_EQ(b.y, 0);
}

TEST_F(PJTest, StructAddFieldFirstTest) {
  S_y a{.y = 42};
  S_xy b{.x = 0xff, .y = 0xff};

  auto [_, enc_size] = transcode<S_y, S_xy>(&a, &b);

  EXPECT_EQ(enc_size, 4);
  EXPECT_EQ(a.y, 42);
  EXPECT_EQ(b.x, 0);
  EXPECT_EQ(b.y, 42);
}

TEST_F(PJTest, NestedStructSameTest) {
  SO x{.s1 = {.x = 2, .y = 3}, .s2 = {.x = 4, .y = 5}};
  SO y = {};

  auto [_, enc_size] = transcode<SO>(&x, &y);

  EXPECT_EQ(enc_size, 24);
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

TEST_F(PJTest, NestedStructAddInnerFieldSmallTest) {
  SOS x{.s1 = {.x = 2}, .s2 = {.x = 3, .y = 4}};
  SO y = {.s1 = {.x = -1, .y = -1}, .s2 = {.x = -1, .y = -1}};

  auto [_, enc_size] = transcode<SOS, SO>(&x, &y);

  EXPECT_EQ(enc_size, 20);
  EXPECT_EQ(y.s1.x, 2);
  EXPECT_EQ(y.s1.y, 0);
  EXPECT_EQ(y.s2.x, 3);
  EXPECT_EQ(y.s2.y, 4);
}

TEST_F(PJTest, NestedStructRemoveInnerFieldLargeTest) {
  TO x{.x = {.x = 2, .y = 3}, .y = {.x = 4, .y = 5}};
  TO2 y = {};

  auto [_, enc_size] = transcode<TO, TO2>(&x, &y);

  EXPECT_EQ(enc_size, 18);
  EXPECT_EQ(y.x.x, 2);
  EXPECT_EQ(y.y.x, 4);
}

TEST_F(PJTest, NestedStructAddInnerFieldLargeTest) {
  TO2 x{.x = {.x = 2}, .y = {.x = 3}};
  TO y = {.x = {.x = -1, .y = -1}, .y = {.x = -1, .y = -1}};

  auto [_, enc_size] = transcode<TO2, TO>(&x, &y);

  EXPECT_EQ(enc_size, 2);
  EXPECT_EQ(y.x.x, 2);
  EXPECT_EQ(y.x.y, 0);
  EXPECT_EQ(y.y.x, 3);
  EXPECT_EQ(y.y.y, 0);
}

TEST_F(PJTest, NestedStructRemoveInnerStruct) {
  UO x{.x = {.x = 1, .y = 2}, .y = {.x = 3, .y = 4}};
  UO2 y = {};

  auto [_, enc_size] = transcode<UO, UO2>(&x, &y);

  EXPECT_EQ(enc_size, 10);
  EXPECT_EQ(y.x.x, 1);
  EXPECT_EQ(y.x.y, 2);
}

TEST_F(PJTest, NestedStructAddInnerStruct) {
  UO2 x = {.x = {.x = 1, .y = 2}};
  UO y{.x = {.x = -1, .y = -1}, .y = {.x = -1, .y = -1}};

  auto [_, enc_size] = transcode<UO2, UO>(&x, &y);

  EXPECT_EQ(enc_size, 5);
  EXPECT_EQ(y.x.x, 1);
  EXPECT_EQ(y.x.y, 2);
  EXPECT_EQ(y.y.x, 0);
  EXPECT_EQ(y.y.y, 0);
}

}  // namespace pj
