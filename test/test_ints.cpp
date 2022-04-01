#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/ints.pj.hpp"

namespace pj {

TEST_F(PJTest, IntSameTest) {
  Int32 x{.i = 1};
  Int32 y{.i = 0};

  Transcode<Int32>(&x, &y);

  EXPECT_EQ(y.i, 1);
}

TEST_F(PJTest, IntSignExtendTest) {
  Int32 x{.i = -1};
  Int64 y{.i = 0};

  Transcode<Int32, Int64>(&x, &y);

  EXPECT_EQ(y.i, -1);
}

TEST_F(PJTest, IntSignedToUnsignedExtendTest) {
  Int32 x{.i = -1};
  UInt64 y{.i = 0};

  Transcode<Int32, UInt64>(&x, &y);

  EXPECT_EQ(y.i, 0xffffffffUL);
}

TEST_F(PJTest, IntUnsignedToSignedExtendTest) {
  UInt32 x{.i = 0xffffffff};
  Int64 y{.i = 0};

  Transcode<Int32, UInt64>(&x, &y);

  EXPECT_EQ(y.i, 0xffffffff);
}

TEST_F(PJTest, IntZeroExtendTest) {
  UInt32 x{.i = 0xffffffff};
  UInt64 y{.i = 0};

  Transcode<UInt32, UInt64>(&x, &y);

  EXPECT_EQ(y.i, 0xffffffff);
}

TEST_F(PJTest, IntTruncTest) {
  Int64 x{.i = -1};
  Int32 y{.i = 0};

  Transcode<Int64, Int32>(&x, &y);

  EXPECT_EQ(x.i, -1);
  EXPECT_EQ(y.i, -1);
}

}  // namespace pj
