#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include "harness.hpp"
#include "test/vectors.pj.hpp"

namespace pj {

TEST_F(PJTest, VectorSame) {
  A a{.vec = {std::array<uint64_t, 5>{1, 2, 3, 4, 5}}};
  A b;

  auto buf = transcode<A>(&a, &b);
  EXPECT_EQ(buf.get(), nullptr);

  EXPECT_EQ(b.vec.size(), a.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    EXPECT_EQ(b.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorOutlineSame) {
  std::vector<uint64_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  A a{.vec = {&values[0], values.size()}};
  A b;

  auto buf = transcode<A>(&a, &b);
  EXPECT_NE(buf.get(), nullptr);

  EXPECT_EQ(b.vec.size(), a.vec.size());
  for (size_t i = 0; i < a.vec.size(); ++i) {
    EXPECT_EQ(b.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorTruncate) {
  std::vector<uint64_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  A a{.vec = {&values[0], values.size()}};
  B b;

  auto buf = transcode<A, B>(&a, &b);
  EXPECT_NE(buf.get(), nullptr);

  EXPECT_EQ(b.vec.size(), 12);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(b.vec[i], a.vec[i]);
  }
}

TEST_F(PJTest, VectorInlineToOutline) {
  A a{.vec = {std::array<uint64_t, 5>{1, 2, 3, 4, 5}}};
  C c;

  auto buf = transcode<A, C>(&a, &c);
  EXPECT_NE(buf.get(), nullptr);

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
  std::array<NestedViewA, 3> arr{NestedViewA{&first[0], first.size()},
                                 NestedViewA{&second[0], second.size()},
                                 NestedViewA{&third[0], third.size()}};

  NestedA a{.vec = {&arr[0], arr.size()}};
  NestedA b;

  auto buf = transcode<NestedA>(&a, &b);
  EXPECT_NE(buf.get(), nullptr);

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
  std::array<NestedViewA, 3> arr{NestedViewA{&first[0], first.size()},
                                 NestedViewA{&second[0], second.size()},
                                 NestedViewA{&third[0], third.size()}};

  NestedA a{.vec = {&arr[0], arr.size()}};
  NestedB b;

  auto buf = transcode<NestedA, NestedB>(&a, &b);
  EXPECT_NE(buf.get(), nullptr);

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
  std::array<char, 9> outline_name{'0', '1', '2', '3', '4', '5', '6', '7', '8'};

  CollectionA a{
      .items = std::array{
          ItemA{.name = std::array{'X'}, .price = 5},
          ItemA{.name = {&outline_name[0], outline_name.size()}, .price = 8},
          ItemA{.name = std::array{'A', 'B', 'C'}, .price = 1}}};
  CollectionB b;

  auto buf = transcode<CollectionA, CollectionB>(&a, &b);
  EXPECT_EQ(buf.get(), nullptr);

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

  std::array alice{'a', 'l', 'i', 'c', 'e'};
  std::array bob{'b', 'o', 'b'};
  std::array charlie{'c', 'h', 'a', 'r', 'l', 'i', 'e'};

  CollectionB b{
      .owners = std::array{OwnerName{&alice[0], alice.size()},
                           OwnerName{&bob[0], bob.size()},
                           OwnerName{&charlie[0], charlie.size()}},
      .items = std::array{
          ItemB{.name = std::array{'X'}, .price = 5, .quantity = 3},
          ItemB{.name = std::array{'0', '1', '2', '3', '4', '5', '6', '7', '8'},
                .price = 5,
                .quantity = 500},
          ItemB{.name = std::array{'A', 'B', 'C'}, .price = 1, .quantity = 1}}};
  CollectionA a;

  auto buf = transcode<CollectionB, CollectionA>(&b, &a);
  EXPECT_NE(buf.get(), nullptr);

  EXPECT_EQ(a.items.size(), b.items.size());
  for (size_t i = 0; i < b.items.size(); ++i) {
    EXPECT_EQ(a.items[i].name, b.items[i].name);
    EXPECT_EQ(a.items[i].price, b.items[i].price);
  }
}

TEST_F(PJTest, VectorOfStructsSame) {
  using OwnerName = decltype(CollectionB::owners[0]);

  std::array alice{'a', 'l', 'i', 'c', 'e'};
  std::array bob{'b', 'o', 'b'};
  std::array charlie{'c', 'h', 'a', 'r', 'l', 'i', 'e'};

  CollectionB b1{
      .owners = std::array{OwnerName{&alice[0], alice.size()},
                           OwnerName{&bob[0], bob.size()},
                           OwnerName{&charlie[0], charlie.size()}},
      .items = std::array{
          ItemB{.name = std::array{'X'}, .price = 5, .quantity = 3},
          ItemB{.name = std::array{'0', '1', '2', '3', '4', '5', '6', '7', '8'},
                .price = 5,
                .quantity = 500},
          ItemB{.name = std::array{'A', 'B', 'C'}, .price = 1, .quantity = 1}}};
  CollectionB b2;

  auto buf = transcode<CollectionB>(&b1, &b2);
  EXPECT_NE(buf.get(), nullptr);

  EXPECT_EQ(b2.items.size(), b1.items.size());
  EXPECT_EQ(b2.owners.size(), b2.owners.size());
  for (size_t i = 0; i < b1.items.size(); ++i) {
    EXPECT_EQ(b2.items[i].name, b1.items[i].name);
    EXPECT_EQ(b2.items[i].price, b1.items[i].price);
    EXPECT_EQ(b2.items[i].quantity, b1.items[i].quantity);
  }
}

}  // namespace pj
