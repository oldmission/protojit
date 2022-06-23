#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"

#include "test/portals.pj.hpp"

namespace pj {

TEST_F(PJTest, PortalPrecompTest) {
  TestPortalPrecomp precomp;
  Int32 x{.i = 1};
  EXPECT_EQ(precomp.size(&x), 4);
  char buf[4];
  precomp.encode(&x, buf);
  Int32 y{.i = 0};
  precomp.decode<void>(buf, &y, {}, {}, nullptr);
  EXPECT_EQ(y.i, 1);
}

TEST_F(PJTest, PortalJitTest) {
  TestPortal jit;
  Int32 x{.i = 1};
  EXPECT_EQ(jit.size(&x), 4);
  char buf[4];
  jit.encode(&x, buf);
  Int32 y{.i = 0};
  jit.decode<void>(buf, &y, {}, {}, nullptr);
  EXPECT_EQ(y.i, 1);
}

TEST_F(PJTest, TwoPortalsPrecompTest) {
  TestSenderPrecomp sender;
  TestReceiverPrecomp receiver;

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

TEST_F(PJTest, TwoPortalsJitTest) {
  TestSender sender;
  TestReceiver receiver;

  auto sender_proto = sender.getProtocol();
  auto receiver_proto = receiver.getProtocol();

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
