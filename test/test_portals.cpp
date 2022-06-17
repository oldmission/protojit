#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/portals.pj.hpp"

namespace pj {

TEST_F(PJTest, PortalPrecompTest) {
  TestPortal::Precomp precomp;
  Int32 x{.i = 1};
  EXPECT_EQ(precomp.size(&x), 4);
  char buf[4];
  precomp.encode(&x, buf);
  Int32 y{.i = 0};
  precomp.decode<void>(buf, &y, {}, {}, nullptr);
  EXPECT_EQ(y.i, 1);
}

}  // namespace pj
