#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "pj/any.hpp"

#include "harness.hpp"
#include "pj/protojit.hpp"

#include "test/ints.pj.hpp"

namespace pj {

TEST_F(PJTest, IntSameTest) {
  Int32 x{.i = 42};
  Any y;

  auto results = transcode(Options<Int32, Any>{.from = &x, .to = &y});

  EXPECT_EQ(y.kind(), Any::Kind::Struct);
  EXPECT_EQ(AnyStruct(y).numFields(), 1);
  EXPECT_EQ(AnyStruct(y).getField(0).name(), "i");
}

}  // namespace pj
