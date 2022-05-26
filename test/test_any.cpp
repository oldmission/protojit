#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/any.pj.hpp"

namespace pj {

TEST_F(PJTest, IntSameTest) {
  Int32 x{.i = 1};
  Any y;

  transcode<Int32, Any>(.from = &x, .to = &y);
}

}  // namespace pj
