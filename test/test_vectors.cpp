#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include "harness.hpp"
#include "test/vectors.pj.hpp"

namespace pj {

TEST_F(PJTest, VectorSame) {
  auto ary = std::array<uint64_t, 5>{1, 2, 3, 4, 5};
  A a{.vec = {ary.data(), ary.size()}};
  A b;

  auto results = transcode(Options<A>{
      .from = &a,
      .to = &b,
      .expect_dec_buffer = true,
  });

  EXPECT_EQ(results.enc_size, /*length=*/1 + /*inline data=*/8 * 8);
  EXPECT_EQ(b.vec.size(), a.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    EXPECT_EQ(b.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorOutlineSame) {
  std::vector<uint64_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  A a{.vec = {values.data(), values.size()}};
  A b;

  auto results =
      transcode(Options<A>{.from = &a, .to = &b, .expect_dec_buffer = true});

  EXPECT_EQ(results.enc_size,
            /*length=*/1 + /*ref=*/8 + /*ppl=*/7 * 8 + /*outline data=*/6 * 8);
  EXPECT_EQ(results.dec_buffer_size, 8 * 13);
  EXPECT_EQ(b.vec.size(), a.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    EXPECT_EQ(b.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorOutlineDifferent) {
  std::vector<uint64_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  B b{.vec = {&values[0], values.size()}};
  A a;

  auto results =
      transcode(Options<B, A>{.from = &b, .to = &a, .expect_dec_buffer = true});

  EXPECT_EQ(a.vec.size(), 12);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(a.vec[i], b.vec[i]);
  }
}

TEST_F(PJTest, VectorTruncate) {
  std::vector<uint64_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  A a{.vec = {values.data(), values.size()}};
  B b;

  auto results =
      transcode(Options<A, B>{.from = &a, .to = &b, .expect_dec_buffer = true});

  EXPECT_EQ(results.dec_buffer_size, 8 * 12);
  EXPECT_EQ(b.vec.size(), 12);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(b.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorTruncateBelowInline) {
  std::vector<uint64_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  A a{.vec = {&values[0], values.size()}};
  D d;

  auto results =
      transcode(Options<A, D>{.from = &a, .to = &d, .expect_dec_buffer = true});

  EXPECT_EQ(d.vec.size(), 7);
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(d.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorToOutline) {
  auto ary = std::array<uint64_t, 5>{1, 2, 3, 4, 5};
  A a{.vec = {ary.data(), ary.size()}};
  C c;

  auto results =
      transcode(Options<A, C>{.from = &a, .to = &c, .expect_dec_buffer = true});

  EXPECT_EQ(results.dec_buffer_size, 8 * 5);
  EXPECT_EQ(c.vec.size(), a.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    EXPECT_EQ(c.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, NestedVectorSame) {
  std::array<uint64_t, 8> first{1, 5, 3, 7, 5, 9, 7, 4};
  std::array<uint64_t, 1> second{42};
  std::array<uint64_t, 5> third{50, 51, 52, 53, 54};

  using NestedViewA = std::decay_t<decltype(NestedA::vec[0])>;
  std::array<NestedViewA, 3> arr{NestedViewA{first.data(), first.size()},
                                 NestedViewA{second.data(), second.size()},
                                 NestedViewA{third.data(), third.size()}};

  NestedA a{.vec = {arr.data(), arr.size()}};
  NestedA b;

  auto results = transcode(
      Options<NestedA>{.from = &a, .to = &b, .expect_dec_buffer = true});

  // Head sizes of various types:
  //   uint64[:]: length (8), ref(8) = 16
  //   uint64[:][2:4]: length (1), ref+ppl (2*16) = 33
  // So the total head size is just 33
  // The outlined data is:
  //   Outline data of partial payload uint64[:]: 8*8 = 64
  //   2 outlined uint64[:]:
  //     head (2*16), outline data (1*8 + 5*8) = 32 + 48 = 80
  EXPECT_EQ(results.enc_size, 33 + 64 + 80);

  // Host types have no partial payload, and both the outer and inner vector
  // exceed the inline length, so the entire vector contents are in the decode
  // buffer.
  EXPECT_EQ(results.dec_buffer_size,
            3 * (/*length of inner=*/8 +
                 /*ref of inner=*/8) +
                /*data for inner vectors=*/8 * (8 + 1 + 5));

  EXPECT_EQ(a.vec.size(), b.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    EXPECT_EQ(b.vec[i].size(), a.vec[i].size());
    for (size_t j = 0; j < a.vec[i].size(); ++j) {
      EXPECT_EQ(a.vec[i][j], b.vec[i][j]);
    }
  }
}

TEST_F(PJTest, NestedVectorDifferent) {
  std::array<uint64_t, 8> first{1, 5, 3, 7, 5, 9, 7, 4};
  std::array<uint64_t, 1> second{42};
  std::array<uint64_t, 5> third{50, 51, 52, 53, 54};

  using NestedViewA = std::decay_t<decltype(NestedA::vec[0])>;
  std::array<NestedViewA, 3> arr{NestedViewA{first.data(), first.size()},
                                 NestedViewA{second.data(), second.size()},
                                 NestedViewA{third.data(), third.size()}};

  NestedA a{.vec = {arr.data(), arr.size()}};
  NestedB b;

  auto results = transcode(Options<NestedA, NestedB>{
      .from = &a, .to = &b, .expect_dec_buffer = true});

  EXPECT_EQ(results.dec_buffer_size, 144);

  EXPECT_EQ(a.vec.size(), b.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    size_t limit = std::min(6ul, a.vec[i].size());
    EXPECT_EQ(b.vec[i].size(), limit);
    for (size_t j = 0; j < limit; ++j) {
      EXPECT_EQ(a.vec[i][j], b.vec[i][j]);
    }
  }
}

TEST_F(PJTest, VectorOfStructsForwards) {
  auto items = std::array{
      ItemA{.name = "X", .price = 5},
      ItemA{.name = "012345678", .price = 8},
      ItemA{.name = "ABC", .price = 1},
  };

  CollectionA a{.items = {items.data(), items.size()}};
  CollectionB b;

  auto results = transcode(Options<CollectionA, CollectionB>{
      .from = &a,
      .to = &b,
      .expect_dec_buffer = true,
  });

  EXPECT_EQ(b.items.size(), a.items.size());
  EXPECT_EQ(b.owners.size(), 0);
  for (size_t i = 0; i < a.items.size(); ++i) {
    EXPECT_EQ(b.items[i].name, a.items[i].name);
    EXPECT_EQ(b.items[i].price, a.items[i].price);
    EXPECT_EQ(b.items[i].quantity, 0);
  }
}

TEST_F(PJTest, VectorOfStructsBackwards) {
  using OwnerName = decltype(CollectionB::owners[0]);

  auto items =
      std::array{ItemB{.name = "X", .price = 5, .quantity = 3},
                 ItemB{.name = "012345678", .price = 5, .quantity = 500},
                 ItemB{.name = "ABC", .price = 1, .quantity = 1}};
  auto owners =
      std::array{OwnerName{"alice"}, OwnerName{"bob"}, OwnerName{"charlie"}};

  CollectionB b{.owners = {owners.data(), owners.size()},
                .items = {items.data(), items.size()}};
  CollectionA a;

  auto results = transcode(Options<CollectionB, CollectionA>{
      .from = &b,
      .to = &a,
      .expect_dec_buffer = true,
  });

  // Head sizes of various types:
  //   ItemB.name = char8[12:]: length (8), inline data (12) = 20
  //   ItemB: name (20), price (4), quantity (4) = 28
  //   char8[:12]: length (1), ref (8) = 9
  //   owners: length (1) + inline data (3*9) = 28
  //   items: length (8) + inline data (5*28) = 148
  // So the total head size is: 28 + 148 = 176
  // The only data that exceeds any inline data is the owner names, which add up
  // to 15 bytes total
  EXPECT_EQ(results.enc_size, 191);
  EXPECT_EQ(results.dec_buffer_size, 85);

  EXPECT_EQ(a.items.size(), b.items.size());
  for (size_t i = 0; i < b.items.size(); ++i) {
    EXPECT_EQ(a.items[i].name, b.items[i].name);
    EXPECT_EQ(a.items[i].price, b.items[i].price);
  }
}

TEST_F(PJTest, VectorOfStructsSame) {
  using OwnerName = decltype(CollectionB::owners[0]);

  auto owners =
      std::array{OwnerName{"ALICE"}, OwnerName{"BOB"}, OwnerName{"CHARLIE"}};
  auto items =
      std::array{ItemB{.name = "X", .price = 5, .quantity = 3},
                 ItemB{.name = "012345678", .price = 5, .quantity = 500},
                 ItemB{.name = "ABC", .price = 1, .quantity = 1}};

  CollectionB b1{.owners = {owners.data(), owners.size()},
                 .items = {items.data(), items.size()}};
  CollectionB b2;

  auto results = transcode(
      Options<CollectionB>{.from = &b1, .to = &b2, .expect_dec_buffer = true});

  EXPECT_EQ(results.dec_buffer_size, 149);
  EXPECT_EQ(b2.items.size(), b1.items.size());
  EXPECT_EQ(b2.owners.size(), b2.owners.size());
  for (size_t i = 0; i < b1.items.size(); ++i) {
    EXPECT_EQ(b2.items[i].name, b1.items[i].name);
    EXPECT_EQ(b2.items[i].price, b1.items[i].price);
    EXPECT_EQ(b2.items[i].quantity, b1.items[i].quantity);
  }
}

TEST_F(PJTest, SizingAlignment) {
  char str[5] = {'A', 'B', 'C', 'D', 'E'};
  wchar_t wstr[5] = {'A', 'B', 'C', 'D', 'E'};

  TestAlignment src{.str = {str, 5}, .wstr = {wstr, 5}};
  TestAlignment dst;

  auto proto = ctx->fromMemory(gen::BuildPJType<TestAlignment>::build(
      ctx->get(), PJGetWireDomain(ctx->get())));

  auto results = transcode(Options<TestAlignment>{
      .from = &src,
      .to = &dst,
      .expect_dec_buffer = true,
      .proto = proto,
  });

  EXPECT_EQ(results.enc_size, 48);
  EXPECT_EQ(src.str.size(), 5);
  EXPECT_EQ(dst.wstr.size(), 5);
}

}  // namespace pj
