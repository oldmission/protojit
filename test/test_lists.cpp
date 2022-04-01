#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "pj/protojit.hpp"

#include "test/lists.pj.hpp"
#include "harness.hpp"

namespace pj {

TEST_F(PJTest, CharListLongTransport) {
  auto sizer0 = GenSize<A1>();

  A1 a0{"abcd", 4};
  EXPECT_EQ(sizer0(&a0), 1 /*size*/ + 8 /*ref*/ + 4 /*chars*/);

  A1 a1{"", 0};
  EXPECT_EQ(sizer0(&a1), 1 /*size*/ + 8 /*ref*/);
}

TEST_F(PJTest, CharList65KTransport) {
  auto sizer1 = GenSize<A1>(1L << 16);

  A1 a2{"abcdefghi", 9};
  EXPECT_EQ(sizer1(&a2), 1 /*size*/ + 2 /*ref*/ + 6 /*chars*/);
}

TEST_F(PJTest, CharList8KTransport) {
  auto sizer2 = GenSize<A1>(8000);

  A1 a2{"abcdefghi", 9};
  EXPECT_EQ(sizer2(&a2), 1 /*size*/ + 2 /*ref*/ + 6 /*chars*/);
}

TEST_F(PJTest, StructList2) {
  auto sizer0 = GenSize<TS1>(1L << 16);

  S1 t0_data[2] = {{.x = 1, .y = 2}, {.x = 3, .y = 4}};
  TS1 t0{t0_data, 2};
  EXPECT_EQ(sizer0(&t0), 1 /*size*/ + 2 /*ref*/ + 24 /*data*/);
}

TEST_F(PJTest, StructList2Longer) {
  auto sizer1 = GenSize<TS1, TS2>(1L << 16);

  S1 t0_data[2] = {{.x = 1, .y = 2}, {.x = 3, .y = 4}};
  TS1 t0{t0_data, 2};
  EXPECT_EQ(sizer1(&t0), 1 /*size*/ + 8 * 12 /*inline data*/);
}

TEST_F(PJTest, StructList10Longer) {
  std::vector<S1> t1_data;
  for (int i = 0; i < 10; ++i) {
    t1_data.emplace_back(S1{.x = i, .y = i + 2});
  }

  TS1 t1{t1_data.data(), 10};

  auto sizer2 = GenSize<TS1, TS2>(1L << 16);
  EXPECT_EQ(sizer2(&t1),
            1 /*size*/ + 8 * 12 /*inline data*/ + 3 * 12 /*outline data*/);
}

TEST_F(PJTest, StructList20Truncate) {
  auto sizer3 = GenSize<TS2, TS1>(1L << 16);

  std::vector<S1> t2_data;
  for (int i = 0; i < 20; ++i) {
    t2_data.emplace_back(S1{.x = i, .y = i + 2});
  }

  TS2 t2{t2_data.data(), 20};

  EXPECT_EQ(sizer3(&t2), 1 /*size*/ + 2 /*ref*/ + 16 * 12 /*outline data*/);
}

TEST_F(PJTest, ListLists) {
  S1 data[2] = {{.x = 1, .y = 2}, {.x = 3, .y = 4}};
  TS1 t1_data[2] = {{data, 2}, {data, 2}};
  TS1L t1{t1_data, 2};

  auto sizer0 = GenSize<TS1L>(1L << 16);
  sizer0(&t1);
  const size_t inner_size = 1 /*size*/ + 2 /*ref*/ + 2 * 12 /*outline data*/;
  EXPECT_EQ(sizer0(&t1),
            1 /*size*/ + 2 /*ref*/ + 2 * inner_size /*outline data*/);
}

#if 0  // SAMIR_TODO
TEST_F(PJTest, VariantLists) {
  V1 data[3] = {
      {
          .value = {.x = {.x = 3, .y = 4}},
          .tag = V1::Kind::x,
      },
      {
          .value = {.y = {.a = 9}},
          .tag = V1::Kind::y,
      },
  };

  TV1 list(data, 3);

  auto sizer0 = GenSize<TV1, PV1>(1L << 16);
  sizer0(&list);
};
#endif

}  // namespace pj
