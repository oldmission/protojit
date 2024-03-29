#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "test/variants.pj.hpp"

namespace pj {

TEST_P(PJVariantTest, VariantSame) {
  Var1 F{Var1::x, 42};
  Var1 T{};

  onMatch<0, Var1>("x", [&](const Var1& T) {
    EXPECT_EQ(T.value.x, 42);
    EXPECT_EQ(T.tag, Var1::Kind::x);
  });

  auto results = transcode(
      Options<Var1>{.from = &F, .to = &T, .src_path = "x", .tag_path = "_"});

  EXPECT_EQ(results.enc_size, 9);
  EXPECT_EQ(F.value.x, 42);
}

TEST_P(PJVariantTest, VariantMismatch) {
  Var1 F(Var1::x, 42);
  Var2 T(Var2::y, -1);

  onNoMatch<0, Var2>("y", [&](const Var2& T) {});
  onMatch<1, Var2>("undef",
                   [&](const Var2& T) { EXPECT_EQ(T.tag, Var2::Kind::undef); });

  transcode(Options<Var1, Var2>{
      .from = &F, .to = &T, .src_path = "x", .tag_path = "_"});
}

TEST_P(PJVariantTest, VariantInvalidHandler) {
  Var1 F{Var1::x, 42};
  Var2 T{};

  onNoMatch<0, Var2>("x", [&](const Var2& T) {});

  transcode(Options<Var1, Var2>{
      .from = &F, .to = &T, .src_path = "x", .tag_path = "_"});
}

TEST_P(PJVariantTest, VariantAddCaseBig) {
  Var1 F{Var1::x, 42};
  Var3 T{};

  onMatch<0, Var3>("x", [&](const Var3& T) {
    EXPECT_EQ(T.value.x, 42);
    EXPECT_EQ(T.tag, Var3::Kind::x);
  });

  transcode(Options<Var1, Var3>{
      .from = &F, .to = &T, .src_path = "x", .tag_path = "_"});
}

TEST_P(PJVariantTest, VariantMissingHandler) {
  Var1 F{Var1::x, 42};
  Var3 T{};

  onNoMatch<0, Var3>("y", [&](const Var3& T) {});
  onNoMatch<0, Var3>("undef", [&](const Var3& T) {});

  transcode(Options<Var1, Var3>{
      .from = &F, .to = &T, .src_path = "x", .tag_path = "_"});
}

TEST_P(PJVariantTest, VariantMoveCase) {
  Var1 F{Var1::x, 42};
  Var4 T{};

  onMatch<0, Var4>("x", [&](const Var4& T) {
    EXPECT_EQ(T.tag, Var4::Kind::x);
    EXPECT_EQ(T.value.x, 42);
  });

  transcode(Options<Var1, Var4>{
      .from = &F, .to = &T, .src_path = "x", .tag_path = "_"});
}

TEST_P(PJVariantTest, VariantMoveCase2) {
  Var3 F{Var3::x, 42};
  Var4 T{};

  onMatch<0, Var4>("x", [&](const Var4& T) {
    EXPECT_EQ(T.tag, Var4::Kind::x);
    EXPECT_EQ(T.value.x, 42);
  });

  auto results = transcode(Options<Var3, Var4>{
      .from = &F, .to = &T, .src_path = "x", .tag_path = "_"});

  EXPECT_EQ(results.enc_size, no_tag ? 65 : 9);
}

TEST_F(PJTest, VariantAddTagField) {
  Outer2 F{.z = 0xab};
  Outer T{.v = {Var4::x, 42}, .z = 0};

  transcode(Options<Outer2, Outer>{.from = &F, .to = &T});

  EXPECT_EQ(T.v.tag, Var4::Kind::undef);
  EXPECT_EQ(T.z, 0xab);
}

TEST_P(PJVariantTest, VariantRemoveTagField) {
  Outer F{.v = {}, .z = 0xab};
  Outer2 T{.z = 0};

  auto results = transcode(Options<Outer, Outer2>{
      .from = &F, .to = &T, .src_path = "v.undef", .tag_path = "v._"});

  EXPECT_EQ(results.enc_size, no_tag ? 10 : 2);
  EXPECT_EQ(T.z, 0xab);
}

TEST_P(PJVariantTest, VariantSameNestedPath) {
  Outer F{.v = {Var4::x, 42}, .z = 0xab};
  Outer T{.v = {}, .z = 0};

  onMatch<0, Outer>("v.x", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::x);
    EXPECT_EQ(T.v.value.x, 42);
    EXPECT_EQ(T.z, 0xab);
  });

  auto results = transcode(Options<Outer>{
      .from = &F, .to = &T, .src_path = "v.x", .tag_path = "v._"});

  EXPECT_EQ(results.enc_size, 10);
}

TEST_F(PJTest, VariantDispatchDefault) {
  Outer2 F{.z = 0xab};
  Outer T{.v = {Var4::w, 33}, .z = 0};

  onMatch<0, Outer>("v.undef", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  transcode(Options<Outer2, Outer>{.from = &F, .to = &T});
}

TEST_P(PJVariantTest, VariantDispatchUndef) {
  Outer F{.v = {}, .z = 0xab};
  Outer T{.v = {Var4::w, 0}, .z = 0};

  onMatch<0, Outer>("v.undef", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  transcode(Options<Outer>{
      .from = &F,
      .to = &T,
      .src_path = "v.undef",
      .tag_path = "v._",
  });
}

TEST_F(PJTest, VariantDispatchDefaultNested) {
  Outer2 F{.z = 0xab};
  NestedOuter T = {};

  onMatch<0, NestedOuter>("p.a.undef", [&](const NestedOuter& T) {
    EXPECT_EQ(T.p.a.tag, Var4::Kind::undef);
    EXPECT_EQ(T.p.b.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  transcode(Options<Outer2, NestedOuter>{.from = &F, .to = &T});
}

TEST_P(PJVariantTest, VariantDifferentDispatchTag) {
  Outer3 F{
      .a = {Var4::w, 0x11},
      .b = {Var4::x, 0x22222222},
  };
  Outer3 T{.a = {}, .b = {}};

  onMatch<0, Outer3>("b.x", [&T](const Outer3& T2) {
    EXPECT_EQ(&T, &T2);
    EXPECT_EQ(T.a.tag, Var4::Kind::w);
    EXPECT_EQ(T.a.value.w, 0x11);
    EXPECT_EQ(T.b.tag, Var4::Kind::x);
    EXPECT_EQ(T.b.value.x, 0x22222222);
  });

  auto results = transcode(Options<Outer3>{
      .from = &F, .to = &T, .src_path = "a.w", .tag_path = "a._"});

  EXPECT_EQ(results.enc_size, no_tag ? 18 : 11);
}

TEST_P(PJVariantTest, VariantAfterVector) {
  std::array<uint64_t, 4> values{1, 2, 3, 4};
  VecVar F{.vec = {values.data(), values.size()}, .var = {Var4::w, 42}};
  VecVar T;

  onMatch<0, VecVar>("var.w", [&](const VecVar& T) {
    EXPECT_EQ(T.var.tag, Var4::Kind::w);
    EXPECT_EQ(T.var.value.w, 42);
    EXPECT_EQ(T.vec, F.vec);
  });

  auto results = transcode(Options<VecVar>{.from = &F,
                                           .to = &T,
                                           .src_path = "var.w",
                                           .tag_path = "var._",
                                           .expect_dec_buffer = true});

  // Vector has 8 length bytes, 8 ref bytes, and 4*8 data bytes
  EXPECT_EQ(results.enc_size, (no_tag ? 9 : 2) + 48);
  EXPECT_EQ(results.dec_buffer_size, 8 * 4);
}

TEST_F(PJTest, EnumTableTest) {
  EnumA F = EnumA::x;
  EnumB T = EnumB::undef;

  auto results = transcode(Options<EnumA, EnumB>{.from = &F, .to = &T});

  EXPECT_EQ(results.enc_size, 1);
  EXPECT_EQ(T, EnumB::x);
}

TEST_F(PJTest, EnumToVariant) {
  EnumA F = EnumA::x;
  NotAnEnum T = {NotAnEnum::x, 42};

  auto results = transcode(Options<EnumA, NotAnEnum>{.from = &F, .to = &T});

  EXPECT_EQ(results.enc_size, 1);
  EXPECT_EQ(T.tag, NotAnEnum::Kind::x);
  EXPECT_EQ(T.value.x, 0);
}

TEST_F(PJTest, VariantToEnum) {
  NotAnEnum F = {NotAnEnum::x, 42};
  EnumA T = EnumA::z;

  auto results = transcode(Options<NotAnEnum, EnumA>{.from = &F, .to = &T});

  EXPECT_EQ(results.enc_size, 2);
  EXPECT_EQ(T, EnumA::x);
}

TEST_P(PJVariantTest, DefaultForwards) {
  DefaultA F{DefaultA::a, MessageA{.id = 42, .number = 22}};
  DefaultAB T{DefaultAB::unknown, Header{.id = 999}};

  auto results = transcode(Options<DefaultA, DefaultAB>{
      .from = &F, .to = &T, .src_path = "a", .tag_path = "_"});

  EXPECT_EQ(T.tag, DefaultAB::Kind::a);
  EXPECT_EQ(T.value.a.id, 42);
  EXPECT_EQ(T.value.a.number, 22);
}

TEST_P(PJVariantTest, DefaultBackwards) {
  DefaultAB F{DefaultAB::b, MessageB{.id = 42, .character = 'A'}};
  DefaultA T{DefaultA::unknown, Header{.id = 33}};

  auto results = transcode(Options<DefaultAB, DefaultA>{
      .from = &F, .to = &T, .src_path = "b", .tag_path = "_"});

  EXPECT_EQ(T.tag, DefaultA::Kind::unknown);
  EXPECT_EQ(T.value.unknown.id, 42);
}

TEST_P(PJVariantTest, EncodeDefault) {
  DefaultA F{DefaultA::unknown, Header{.id = 42}};
  DefaultAB T{DefaultAB::unknown, Header{.id = 66}};

  auto results = transcode(Options<DefaultA, DefaultAB>{
      .from = &F, .to = &T, .src_path = "unknown", .tag_path = "_"});

  EXPECT_EQ(T.tag, DefaultAB::Kind::unknown);
  EXPECT_EQ(T.value.a.id, 42);
}

// Automatically tests all combinations of src_path and tag_path being provided
// or not being provided.
INSTANTIATE_TEST_SUITE_P(Variants, PJVariantTest,
                         testing::Values(std::make_pair(false, false),
                                         std::make_pair(false, true),
                                         std::make_pair(true, false),
                                         std::make_pair(true, true)));

}  // namespace pj
