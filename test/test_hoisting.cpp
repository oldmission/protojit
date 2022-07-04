#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include "harness.hpp"
#include "test/hoisting.pj.hpp"

namespace pj {

TEST_P(PJVariantTest, HoistingInline) {
  std::array<long, 5> name{1, 2, 3, 4, 5};

  using Name = decltype(FirstInner::name);

  A a{A::f(), First{.x = {.name = Name{&name[0], name.size()}}}};
  A b{};

  auto results = transcode(Options<A>{
      .from = &a,
      .to = &b,
      .tag_path = "_",
      .expect_dec_buffer = true,
  });

  // Length is 1 byte because hoisting makes the max size the same as the inline
  // size.
  EXPECT_EQ(results.enc_size, /*tag=*/1 + /*length of f_short=*/1 +
                                  /*inline data for f_short=*/64);
  EXPECT_EQ(b.tag, A::Kind::f);
  EXPECT_EQ(b.value.f.x.name, a.value.f.x.name);
}

TEST_P(PJVariantTest, HoistingOutline) {
  std::array<long, 10> name{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  using Name = decltype(FirstInner::name);

  A a{A::f(), First{.x = {.name = Name{&name[0], name.size()}}}};
  A b{};

  auto results = transcode(Options<A>{
      .from = &a, .to = &b, .tag_path = "_", .expect_dec_buffer = true});

  if (no_tag) {
    EXPECT_EQ(results.enc_size,
              /*tag=*/1 + /*inline variant storage (wasted)=*/(1 + 64) +
                  /*outline data=*/10 * 8);
  } else {
    // Length is 2 bytes in this case because the max_size is 256 and the
    // outline version is used.
    EXPECT_EQ(results.enc_size, /*tag=*/1 + /*length of f_long=*/2 +
                                    /*ref of f_long=*/8 +
                                    /*outline data=*/10 * 8);
  }
  EXPECT_EQ(b.tag, A::Kind::f);
  EXPECT_EQ(b.value.f.x.name, a.value.f.x.name);
}

TEST_P(PJVariantTest, MultipleHoistingDifferentTermInline) {
  std::array<long, 4> name{1, 2, 3, 4};

  using Name = decltype(SecondInner::name);

  B a{B::s(), Second{.x = {.name = Name{&name[0], name.size()}}}};
  B b{};

  auto results = transcode(Options<B>{
      .from = &a,
      .to = &b,
      .tag_path = "_",
      .expect_dec_buffer = true,
  });

  if (no_tag) {
    // f_short is the largest of the terms in the inline variant.
    EXPECT_EQ(
        results.enc_size,
        /*tag=*/1 + /*length of f_short=*/1 + /*inline data for f_short*/ 64);
  } else {
    EXPECT_EQ(results.enc_size, /*tag=*/1 + /*length of s_short=*/1 +
                                    /*inline data for s_short=*/32);
  }
  EXPECT_EQ(b.tag, B::Kind::s);
  EXPECT_EQ(b.value.s.x.name, a.value.s.x.name);
}

TEST_P(PJVariantTest, MultipleHoistingDifferentTermOutline) {
  std::array<long, 10> name{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  using Name = decltype(SecondInner::name);

  B a{B::s(), Second{.x = {.name = Name{&name[0], name.size()}}}};
  B b;

  auto results = transcode(Options<B>{
      .from = &a, .to = &b, .tag_path = "_", .expect_dec_buffer = true});

  if (no_tag) {
    // f_short is the largest term in the inline variant, with a size of 65.
    EXPECT_EQ(
        results.enc_size,
        /*tag=*/1 + /*inline variant storage=*/65 + /*outline data=*/10 * 8);
  } else {
    // Length is 8 bytes because the outline version is used and max_length is
    // not set.
    EXPECT_EQ(results.enc_size, /*tag=*/1 + /*length of s_long=*/8 +
                                    /*ref of s_long=*/8 +
                                    /*outline data=*/10 * 8);
  }
  EXPECT_EQ(b.tag, B::Kind::s);
  EXPECT_EQ(b.value.s.x.name, a.value.s.x.name);
}

TEST_P(PJVariantTest, MultipleHoistingSameTermInline) {
  std::array<long, 4> first_name{1, 2, 3, 4};
  std::array<long, 4> second_name{5, 6, 7, 8};

  using FirstName = decltype(FirstInner::name);
  using SecondName = decltype(SecondInner::name);

  C a{C::t(),
      Third{.x = {.name = FirstName{&first_name[0], first_name.size()}},
            .y = {.name = SecondName{&second_name[0], second_name.size()}}}};
  C b{};

  auto results = transcode(Options<C>{
      .from = &a,
      .to = &b,
      .tag_path = "_",
      .expect_dec_buffer = true,
  });

  // The largest term in C is t, which contains FirstInner (x) and SecondInner
  // (y). The inline and outline versions are the same since the vectors are the
  // same either way.
  EXPECT_EQ(results.enc_size,
            /*tag=*/1 + /*x_short=*/(1 + 64) + /*y_short=*/(1 + 32));
  EXPECT_EQ(b.tag, C::Kind::t);
  EXPECT_EQ(b.value.t.x.name, a.value.t.x.name);
  EXPECT_EQ(b.value.t.y.name, a.value.t.y.name);
}

TEST_P(PJVariantTest, MultipleHoistingSameTermPartialOutline) {
  std::array<long, 4> first_name{1, 2, 3, 4};
  std::array<long, 5> second_name{5, 6, 7, 8, 9};

  using FirstName = decltype(FirstInner::name);
  using SecondName = decltype(SecondInner::name);

  C a{C::t(),
      Third{.x = {.name = FirstName{&first_name[0], first_name.size()}},
            .y = {.name = SecondName{&second_name[0], second_name.size()}}}};
  C b{};

  auto results = transcode(Options<C>{
      .from = &a, .to = &b, .tag_path = "_", .expect_dec_buffer = true});

  if (no_tag) {
    // The inline variant storage is equal to the size of x_short + y_short. The
    // size of y_long is 8 (length) + 8 (ref) which is smaller than 1 + 32.
    EXPECT_EQ(results.enc_size,
              /*tag=*/1 + /*inline variant storage (x_short + y_short)=*/
                  ((1 + 64) + (1 + 32)) + /*outline data of y_long=*/5 * 8);
  } else {
    EXPECT_EQ(results.enc_size,
              /*tag=*/1 + /*x_short=*/(1 + 64) + /*length of s_long=*/8 +
                  /*ref of s_long=*/8 + /*outline data of s_long=*/5 * 8);
  }
  EXPECT_EQ(b.tag, C::Kind::t);
  EXPECT_EQ(b.value.t.x.name, a.value.t.x.name);
  EXPECT_EQ(b.value.t.y.name, a.value.t.y.name);
}

// Test with and without tag path.
INSTANTIATE_TEST_SUITE_P(HoistingVariants, PJVariantTest,
                         testing::Values(std::make_pair(false, false),
                                         std::make_pair(true, false)));

}  // namespace pj
