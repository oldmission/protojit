#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include "harness.hpp"
#include "test/hoisting.pj.hpp"

namespace pj {

TEST_F(PJTest, Hoisting) {
  std::array<long, 18> name;

  using Name = decltype(Second::name);

  A a{.value = {.f = {.x = {.name = Name{&name[0], name.size()}}}},
      .tag = A::Kind::f};
  A dest{.value = {.f = {.x = {.name = {}}}}, .tag = A::Kind::f};

  auto [_, enc_size] = transcode<A, A, AProto>(&a, &dest);
}  // namespace pj

}  // namespace pj
