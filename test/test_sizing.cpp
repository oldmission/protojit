#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include "harness.hpp"
#include "test/sizing.pj.hpp"

namespace pj {

TEST_F(PJTest, VectorOfStructsRoundUp) {
  std::array<uint64_t, 16> long_v1;
  std::array<char, 10> long_v2;
  A long_A{.v1 = {long_v1.data(), long_v1.size()},
           .v2 = {long_v2.data(), long_v2.size()}};

  std::array<A, 10> long_v3;
  long_v3.fill(long_A);

  B b{.v3 = {long_v3.data(), long_v3.size()}};

  auto results = transcode(Options<B>{.from = &b, .round_up_size = true});

  // Because all of the vectors are filled to their maximum lengths, we expect
  // that the rounded up size is EQUAL to the actual size of the vector. Manual
  // inspection of the generated code shows that the function actually returns a
  // constant value.

  // Head sizes of various types:
  //   uint64[8:16]: length (1), ref (8), ppl (7*8) = 65
  //   char8[2:10]: length (1), ref (8) = 9
  //   A: 65 + 9 = 74
  //   A[1:10]: length (1), ref+ppl / inline data (1*74) = 75
  // Outline data:
  //   10 uint64[8:16]: 10 * outlined data (9*8) = 720
  //   10 char[2:10]: 10 * outlined data (10) = 100
  //   10 A: 10*74 = 740
  EXPECT_EQ(results.enc_size, 75 + 720 + 100 + 740);
}  // namespace pj

}  // namespace pj
