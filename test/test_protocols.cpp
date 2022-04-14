#include <gtest/gtest.h>

#include "harness.hpp"
#include "test/protocols.pj.hpp"
#include "test/protocols2.pj.hpp"

namespace pj {

TEST_P(PJVariantTest, ProtocolSame) {
  using namespace v1;

  Adoption F = {
      .animal = {.specifics = {.value = {.dog = {.breed = DogBreed::BEAGLE}},
                               .tag = Specifics::Kind::dog},
                 .age = 8,
                 .weight = 40,
                 .gender = 0},
      .date = {.year = 2022, .month = 4, .date = 14},
      .fee = 65};
  Adoption T = {};

  onMatch<0, Adoption>("animal.specifics.dog", [&](const Adoption& A) {
    EXPECT_EQ(A.animal.specifics.value.dog.breed,
              F.animal.specifics.value.dog.breed);
    EXPECT_EQ(A.animal.age, F.animal.age);
    EXPECT_EQ(A.animal.weight, F.animal.weight);
    EXPECT_EQ(A.animal.gender, F.animal.gender);
    EXPECT_EQ(A.date.year, F.date.year);
    EXPECT_EQ(A.date.month, F.date.month);
    EXPECT_EQ(A.date.date, F.date.date);
    EXPECT_EQ(A.fee, F.fee);
  });

  transcode<Adoption, Adoption, AdoptionProto>(&F, &T, "animal.specifics.dog");
}

TEST_P(PJVariantTest, ProtocolForwards) {
  v1::Adoption F = {
      .animal = {.specifics = {.value = {.cat = {.coat = v1::CatCoat::SHORT}},
                               .tag = v1::Specifics::Kind::cat},
                 .age = 12,
                 .weight = 10,
                 .gender = 0},
      .date = {.year = 2022, .month = 4, .date = 14},
      .fee = 65};
  v2::Adoption T = {};

  onMatch<0, v2::Adoption>("animal.specifics.cat", [&](const v2::Adoption& A) {
    EXPECT_EQ(A.location, v2::Location::undef);
    EXPECT_EQ(A.animal.specifics.value.cat.personality,
              v2::CatPersonality::undef);
    EXPECT_EQ(A.animal.specifics.value.cat.coat, v2::CatCoat::SHORT);
    EXPECT_EQ(A.animal.age, F.animal.age);
    EXPECT_EQ(A.animal.weight, F.animal.weight);
    EXPECT_EQ(A.animal.gender, F.animal.gender);
    EXPECT_EQ(A.date.year, F.date.year);
    EXPECT_EQ(A.date.month, F.date.month);
    EXPECT_EQ(A.date.date, F.date.date);
    EXPECT_EQ(A.fee, F.fee);
  });

  transcode<v1::Adoption, v2::Adoption, v1::AdoptionProto>(
      &F, &T, "animal.specifics.cat");
}

TEST_P(PJVariantTest, ProtocolBackwards) {
  v2::Adoption F = {
      .location = v2::Location::SHELTER_C,
      .animal = {.specifics = {.value = {.cat = {.personality =
                                                     v2::CatPersonality::BOSSY,
                                                 .coat = v2::CatCoat::SHORT}},
                               .tag = v2::Specifics::Kind::cat},
                 .age = 12,
                 .weight = 10,
                 .gender = 0},
      .date = {.year = 2022, .month = 4, .date = 14},
      .fee = 0x123};
  v1::Adoption T = {};

  onMatch<0, v1::Adoption>("animal.specifics.cat", [&](const v1::Adoption& A) {
    EXPECT_EQ(A.animal.specifics.value.cat.coat, v1::CatCoat::SHORT);
    EXPECT_EQ(A.animal.age, F.animal.age);
    EXPECT_EQ(A.animal.weight, F.animal.weight);
    EXPECT_EQ(A.animal.gender, F.animal.gender);
    EXPECT_EQ(A.date.year, F.date.year);
    EXPECT_EQ(A.date.month, F.date.month);
    EXPECT_EQ(A.date.date, F.date.date);
    EXPECT_EQ(A.fee, 0x23);
  });

  transcode<v2::Adoption, v1::Adoption, v2::AdoptionProto>(
      &F, &T, "animal.specifics.cat");
}

INSTANTIATE_TEST_SUITE_P(ProtocolVariants, PJVariantTest,
                         testing::Values(std::make_pair(false, false),
                                         std::make_pair(false, true)));

}  // namespace pj
