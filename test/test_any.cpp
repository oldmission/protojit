#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/any.pj.hpp"

namespace pj {

TEST_F(PJTest, IntSameTest) {
  Int32 x{.i = 1};
  Any y;

  auto results = transcode(Options<Int32, Any>{.from = &x, .to = &y});

  // TODO: use offsets instead of pointers in the type representation.
  // Pointers don't work because the string constant gets moved around
  // during compilation.
  // TODO: the data pointer isn't getting set for some reason.
  asm("int3");
  (void)y;
}

}  // namespace pj
