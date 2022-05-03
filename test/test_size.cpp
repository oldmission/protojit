#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "test/size.pj.hpp"

namespace pj {

TEST_F(PJTest, StructSizing) {
  S_xy a{.x = 42, .y = 43};

  EXPECT_EQ(get_size<S_xy>(&a), 12);
}

TEST_F(PJTest, VariantSizingX) {
  Var4 x{.value = {.x = 42}, .tag = Var4::Kind::x};
  EXPECT_EQ(get_size<Var4>(&x, "", "_"), 9);
}

TEST_F(PJTest, VariantSizingW) {
  Var4 w{.value = {.w = 31}, .tag = Var4::Kind::w};
  EXPECT_EQ(get_size<Var4>(&w, "", "_"), 2);
}

TEST_F(PJTest, VariantAfterVectorSizing) {
  std::array<uint64_t, 4> values{1, 2, 3, 4};
  VecVar a{.vec = {&values[0], values.size()},
           .var = {.value = {.w = 42}, .tag = Var4::Kind::w}};

  // vec: length (8 bytes), ref (8 bytes), outline data (4*8 bytes) = 48 bytes
  // Variant: 1 tag byte, 1 data byte = 2 bytes
  EXPECT_EQ(get_size<VecVar>(&a, "", "var._"), 50);
}

TEST_F(PJTest, VectorOfStructsSizingFirst) {
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

  // Inline data sizes of various types:
  //  ItemB.name: length (8), inline data (12) = 20
  //  ItemB: name (20), price (4), quantity (4) = 28
  //  char8[:12]: length (1), ref (8) = 9
  //  owners: length (8) + inline data (3*9) = 35
  //  items: length (8) + inline data (5*28) = 148
  // So the total head size is: 35 + 148 = 183
  // The only data that exceeds any min_length is the owner names, which add up
  // to 15 bytes total, for an expected total of 183 + 15 = 198
  EXPECT_EQ(get_size<CollectionB>(&b, "", ""), 198);
}

TEST_F(PJTest, VectorOfStructsSizingSecond) {
  using OwnerName = decltype(CollectionB::owners[0]);

  std::array alice{'a', 'l', 'i', 'c', 'e'};
  std::array bob{'b', 'o', 'b'};
  std::array charlie{'c', 'h', 'a', 'r', 'l', 'i', 'e'};
  std::array david{'d', 'a', 'v', 'i', 'd'};

  std::array owners{OwnerName{&alice[0], alice.size()},
                    OwnerName{&bob[0], bob.size()},
                    OwnerName{&charlie[0], charlie.size()},
                    OwnerName{&david[0], david.size()}};

  std::array long_item_name{'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                            'S', 'T', 'U', 'V', 'W', 'X'};

  std::array items{
      ItemB{.name = std::array{'X'}, .price = 5, .quantity = 3},
      ItemB{.name = std::array{'0', '1', '2', '3', '4', '5', '6', '7', '8'},
            .price = 5,
            .quantity = 500},
      ItemB{.name = std::array{'A', 'B', 'C'}, .price = 1, .quantity = 1},
      ItemB{.name = std::array{'D', 'E', 'F'}, .price = 2, .quantity = 2},
      ItemB{.name = std::array{'G', 'H', 'I'}, .price = 3, .quantity = 3},
      ItemB{.name = {&long_item_name[0], long_item_name.size()},
            .price = 4,
            .quantity = 4}};

  CollectionB b{.owners = {&owners[0], owners.size()},
                .items = {&items[0], items.size()}};

  // Head size of CollectionB is the same as computed above: 183
  //  There is one ItemB above min_length of 5
  //  There is one item name above min_length of 12
  //  There is one owner name above min_length of 3
  // The outlined data is:
  //  2 ItemBs (one overflow because of the 8 bytes for ref): 2*28
  //  1 ItemB.name: 9 bytes, because 4 remain partially inlined
  //  2 owner names / char8[:12] (one overflow because of ref): 2*9
  //  Owner names: 20 (5 + 3 + 7 + 5)
  // Giving a total of 183 + 2*28 + 9 + 2*9 + 20 = 286
  EXPECT_EQ(get_size<CollectionB>(&b, "", ""), 286);
}

}  // namespace pj
