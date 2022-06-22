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

TEST_F(PJTest, TwoPortalsPrecompTest) {
  TestSender::Precomp sender;
  TestReceiver::Precomp receiver;

  auto sender_proto = ctx->decodeProto(sender.getSchema().data());
  auto receiver_proto = ctx->decodeProto(receiver.getSchema().data());

  EXPECT_TRUE(sender_proto.isBinaryCompatibleWith(receiver_proto));

  Int32 x{.i = 1};
  EXPECT_EQ(sender.size(&x), 4);
  char buf[4];
  sender.encode(&x, buf);
  Int32 y{.i = 0};
  receiver.decode<void>(buf, &y, {}, {}, nullptr);
  EXPECT_EQ(y.i, 1);
}

}  // namespace pj
