#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "test/variants.pj.hpp"

namespace pj {

TEST_P(PJVariantTest, VariantSame) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var1 T{.value = {.x = 0xffffffff}, .tag = static_cast<Var1::Kind>(0x7f)};

  onMatch<0, Var1>("x", [&](const Var1& T) {
    EXPECT_EQ(T.value.x, 42);
    EXPECT_EQ(T.tag, Var1::Kind::x);
  });

  auto [_, enc_size] = transcode<Var1>(&F, &T, "x", "_");

  EXPECT_EQ(enc_size, 9);
  EXPECT_EQ(F.value.x, 42);
}

TEST_P(PJVariantTest, VariantMismatch) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var2 T{.value = {.y = -1}, .tag = Var2::Kind::y};

  onNoMatch<0, Var2>("y", [&](const Var2& T) {});
  onMatch<1, Var2>("undef",
                   [&](const Var2& T) { EXPECT_EQ(T.tag, Var2::Kind::undef); });

  transcode<Var1, Var2>(&F, &T, "x", "_");
}

TEST_P(PJVariantTest, VariantInvalidHandler) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var2 T{.value = {.y = 0}, .tag = Var2::Kind::undef};

  onNoMatch<0, Var2>("x", [&](const Var2& T) {});

  transcode<Var1, Var2>(&F, &T, "x", "_");
}

TEST_P(PJVariantTest, VariantAddCaseBig) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var3 T{.value = {.x = 0}, .tag = Var3::Kind::undef};

  onMatch<0, Var3>("x", [&](const Var3& T) {
    EXPECT_EQ(T.value.x, 42);
    EXPECT_EQ(T.tag, Var3::Kind::x);
  });

  transcode<Var1, Var3>(&F, &T, "x", "_");

  EXPECT_EQ(F.value.x, 42);
}

TEST_P(PJVariantTest, VariantMissingHandler) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var3 T{.value = {.x = 0}, .tag = Var3::Kind::undef};

  onNoMatch<0, Var3>("y", [&](const Var3& T) {});
  onNoMatch<0, Var3>("undef", [&](const Var3& T) {});

  transcode<Var1, Var3>(&F, &T, "x", "_");
}

TEST_P(PJVariantTest, VariantMoveCase) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var4 T{.value = {.x = -1}, .tag = Var4::Kind::undef};

  onMatch<0, Var4>("x", [&](const Var4& T) {
    EXPECT_EQ(T.tag, Var4::Kind::x);
    EXPECT_EQ(T.value.x, 42);
  });

  transcode<Var1, Var4>(&F, &T, "x", "_");
}

TEST_P(PJVariantTest, VariantMoveCase2) {
  Var3 F{.value = {.x = 42}, .tag = Var3::Kind::x};
  Var4 T{.value = {.x = -1}, .tag = Var4::Kind::undef};

  onMatch<0, Var4>("x", [&](const Var4& T) {
    EXPECT_EQ(T.tag, Var4::Kind::x);
    EXPECT_EQ(T.value.x, 42);
  });

  auto [_, enc_size] = transcode<Var3, Var4>(&F, &T, "x", ".");

  EXPECT_EQ(enc_size, no_tag ? 65 : 9);
}

TEST_F(PJTest, VariantAddTagField) {
  Outer2 F{.z = 0xab};
  Outer T{.v = {.value = {.x = 42}, .tag = Var4::Kind::x}, .z = 0};

  transcode<Outer2, Outer>(&F, &T);

  EXPECT_EQ(T.v.tag, Var4::Kind::undef);
  EXPECT_EQ(T.z, 0xab);
}

TEST_P(PJVariantTest, VariantRemoveTagField) {
  Outer F{.v = {.tag = Var4::Kind::undef}, .z = 0xab};
  Outer2 T{.z = 0};

  auto [_, enc_size] = transcode<Outer, Outer2>(&F, &T, "v.undef", "v._");

  EXPECT_EQ(enc_size, no_tag ? 10 : 2);
  EXPECT_EQ(T.z, 0xab);
}

TEST_P(PJVariantTest, VariantSameNestedPath) {
  Outer F{.v = Var4{.value = {.x = 42}, .tag = Var4::Kind::x}, .z = 0xab};
  Outer T{.v = Var4{.value = {.x = -1}, .tag = Var4::Kind::undef}, .z = 0};

  onMatch<0, Outer>("v.x", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::x);
    EXPECT_EQ(T.v.value.x, 42);
    EXPECT_EQ(T.z, 0xab);
  });

  auto [_, enc_size] = transcode<Outer>(&F, &T, "v.x", "v._");

  EXPECT_EQ(enc_size, 10);
}

TEST_F(PJTest, VariantDispatchDefault) {
  Outer2 F{.z = 0xab};
  Outer T{.v = {.tag = Var4::Kind::w}, .z = 0};

  onMatch<0, Outer>("v.undef", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  transcode<Outer2, Outer>(&F, &T);
}

TEST_P(PJVariantTest, VariantDispatchUndef) {
  Outer F{.v = {.tag = Var4::Kind::undef}, .z = 0xab};
  Outer T{.v = {.value = {.w = 0}, .tag = Var4::Kind::w}, .z = 0};

  onMatch<0, Outer>("v.undef", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  transcode<Outer>(&F, &T, "v.undef", "v._");
}

TEST_F(PJTest, VariantDispatchDefaultNested) {
  Outer2 F{.z = 0xab};
  NestedOuter T = {};

  onMatch<0, NestedOuter>("p.a.undef", [&](const NestedOuter& T) {
    EXPECT_EQ(T.p.a.tag, Var4::Kind::undef);
    EXPECT_EQ(T.p.b.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  transcode<Outer2, NestedOuter>(&F, &T);
}

TEST_P(PJVariantTest, VariantDifferentDispatchTag) {
  Outer3 F{
      .a = {.value = {.w = 0x11}, .tag = Var4::Kind::w},
      .b = {.value = {.x = 0x22222222}, .tag = Var4::Kind::x},
  };
  Outer3 T{
      .a = {.tag = Var4::Kind::undef},
      .b = {.tag = Var4::Kind::undef},
  };

  onMatch<0, Outer3>("b.x", [&T](const Outer3& T2) {
    EXPECT_EQ(&T, &T2);
    EXPECT_EQ(T.a.tag, Var4::Kind::w);
    EXPECT_EQ(T.a.value.w, 0x11);
    EXPECT_EQ(T.b.tag, Var4::Kind::x);
    EXPECT_EQ(T.b.value.x, 0x22222222);
  });

  auto [_, enc_size] = transcode<Outer3>(&F, &T, "a.w", "a._");

  EXPECT_EQ(enc_size, no_tag ? 18 : 11);
}

TEST_P(PJVariantTest, VariantAfterVector) {
  std::array<uint64_t, 4> values{1, 2, 3, 4};
  VecVar F{.vec = {&values[0], values.size()},
           .var = {.value = {.w = 42}, .tag = Var4::Kind::w}};
  VecVar T;

  onMatch<0, VecVar>("var.w", [&](const VecVar& T) {
    EXPECT_EQ(T.var.tag, Var4::Kind::w);
    EXPECT_EQ(T.var.value.w, 42);
    EXPECT_EQ(T.vec, F.vec);
  });

  auto [_, enc_size] = transcode<VecVar>(&F, &T, "var.w", "var._");

  // Vector has 8 length bytes, 8 ref bytes, and 4*8 data bytes
  EXPECT_EQ(enc_size, (no_tag ? 9 : 2) + 48);
}

TEST_F(PJTest, EnumTableTest) {
  EnumA F{.tag = EnumA::Kind::x};
  EnumB T{.tag = EnumB::Kind::undef};

  auto [_, enc_size] = transcode<EnumA, EnumB>(&F, &T);

  EXPECT_EQ(enc_size, 1);
  EXPECT_EQ(T.tag, EnumB::Kind::x);
}

// Automatically tests all combinations of src_path and tag_path being provided
// or not being provided.
INSTANTIATE_TEST_SUITE_P(Variants, PJVariantTest,
                         testing::Values(std::make_pair(false, false),
                                         std::make_pair(false, true),
                                         std::make_pair(true, false),
                                         std::make_pair(true, true)));

}  // namespace pj
