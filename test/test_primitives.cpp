#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <cmath>
#include <functional>
#include <limits>

#include "harness.hpp"

#include "test/primitives.pj.hpp"

namespace pj {

TEST_F(PJTest, IntSameTest) {
  Int32 x{.i = 1};
  Int32 y{.i = 0};

  transcode(Options<Int32>{.from = &x, .to = &y});

  EXPECT_EQ(y.i, 1);
}

TEST_F(PJTest, IntSignExtendTest) {
  Int32 x{.i = -1};
  Int64 y{.i = 0};

  transcode(Options<Int32, Int64>{.from = &x, .to = &y});

  EXPECT_EQ(y.i, -1);
}

TEST_F(PJTest, IntSignedToUnsignedExtendTest) {
  Int32 x{.i = -1};
  UInt64 y{.i = 0};

  transcode(Options<Int32, UInt64>{.from = &x, .to = &y});

  EXPECT_EQ(y.i, 0xffffffffffffffffUL);
}

TEST_F(PJTest, IntUnsignedToSignedExtendTest) {
  UInt32 x{.i = 0xffffffff};
  Int64 y{.i = 0};

  transcode(Options<UInt32, Int64>{.from = &x, .to = &y});

  EXPECT_EQ(y.i, 0xffffffffUL);
}

TEST_F(PJTest, IntZeroExtendTest) {
  UInt32 x{.i = 0xffffffff};
  UInt64 y{.i = 0};

  transcode(Options<UInt32, UInt64>{.from = &x, .to = &y});

  EXPECT_EQ(y.i, 0xffffffff);
}

TEST_F(PJTest, IntTruncTest) {
  Int64 x{.i = -1};
  Int32 y{.i = 0};

  transcode(Options<Int64, Int32>{&x, &y});

  EXPECT_EQ(x.i, -1);
  EXPECT_EQ(y.i, -1);
}

TEST_F(PJTest, FloatSameTest) {
  Float32 x{.f = 1.25};
  Float32 y{.f = 0};

  transcode(Options<Float32>{.from = &x, .to = &y});

  EXPECT_EQ(y.f, 1.25);
}

TEST_F(PJTest, FloatExtendTest) {
  Float32 x{.f = 1.25};
  Float64 y{.f = 0};

  transcode(Options<Float32, Float64>{.from = &x, .to = &y});

  EXPECT_EQ(y.f, 1.25);
}

TEST_F(PJTest, FloatTruncTestSimple) {
  Float64 x{.f = 1.25};
  Float32 y{.f = 0};

  transcode(Options<Float64, Float32>{&x, &y});

  EXPECT_EQ(y.f, 1.25);
}

TEST_F(PJTest, FloatTruncTestInfinite) {
  Float64 x{.f = std::numeric_limits<double>::max()};
  Float32 y{.f = 0};

  transcode(Options<Float64, Float32>{&x, &y});

  EXPECT_EQ(y.f, std::numeric_limits<float>::infinity());
}

TEST_F(PJTest, Float32DefaultTest) {
  Empty x{};
  Float32 y{.f = 0};

  transcode(Options<Empty, Float32>{&x, &y});

  EXPECT_TRUE(std::isnan(y.f));
}

TEST_F(PJTest, Float64DefaultTest) {
  Empty x{};
  Float64 y{.f = 0};

  transcode(Options<Empty, Float64>{&x, &y});

  EXPECT_TRUE(std::isnan(y.f));
}

}  // namespace pj
