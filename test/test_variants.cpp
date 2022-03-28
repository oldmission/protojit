#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "test/variants.pj.hpp"

namespace pj {

TEST_F(PJTest, VariantSame) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var1 T{.value = {.x = 0}, .tag = Var1::Kind::undef};

  OnMatch<0, Var1>("x", [&](const Var1& T) {
    EXPECT_EQ(T.value.x, 42);
    EXPECT_EQ(T.tag, Var1::Kind::x);
  });

  Transcode<Var1>(&F, &T, {"x"}, {"."}, {"."});

  EXPECT_EQ(F.value.x, 42);
}

TEST_F(PJTest, VariantSameNoTag) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var1 T{.value = {.x = 0xffffffff}, .tag = static_cast<Var1::Kind>(0x7f)};

  Transcode<Var1>(&F, &T, {"x"});

  EXPECT_EQ(F.tag, Var1::Kind::x);
  EXPECT_EQ(T.value.x, 42);
  EXPECT_EQ(T.tag, Var1::Kind::x);
}

TEST_F(PJTest, VariantSameNoPath) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var1 T{.value = {.x = 0xffffffff}, .tag = static_cast<Var1::Kind>(0x7f)};

  Transcode<Var1>(&F, &T);

  EXPECT_EQ(F.tag, Var1::Kind::x);
  EXPECT_EQ(T.value.x, 42);
  EXPECT_EQ(T.tag, Var1::Kind::x);
}

TEST_F(PJTest, VariantMismatchNoTag) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var2 T{.value = {.y = -1}, .tag = Var2::Kind::y};

  OnNoMatch<0, Var2>("y", [&](const Var2& T) {});

  Transcode<Var1, Var2>(&F, &T, {"x"});

  EXPECT_EQ(T.tag, Var2::Kind::undef);
}

TEST_F(PJTest, VariantMismatch) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var2 T{.value = {.y = -1}, .tag = Var2::Kind::y};

  OnNoMatch<0, Var2>("y", [&](const Var2& T) {});
  OnMatch<0, Var2>("undef", [&](const Var2& T) {});

  Transcode<Var1, Var2>(&F, &T, {"x"}, {"."}, {"."});

  EXPECT_EQ(T.tag, Var2::Kind::undef);
}

TEST_F(PJTest, VariantAddCaseBigNoTag) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var3 T{.value = {.x = -1}, .tag = Var3::Kind::undef};

  Transcode<Var1, Var3>(&F, &T, {"x"});

  EXPECT_EQ(T.tag, Var3::Kind::x);
  EXPECT_EQ(T.value.x, 42);
}

TEST_F(PJTest, VariantAddCaseBig) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var3 T{.value = {.x = 0}, .tag = Var3::Kind::undef};

  OnMatch<0, Var3>("x", [&](const Var3& T) {
    EXPECT_EQ(T.value.x, 42);
    EXPECT_EQ(T.tag, Var3::Kind::x);
  });

  Transcode<Var1, Var3>(&F, &T, {"x"}, {"."}, {"."});

  EXPECT_EQ(F.value.x, 42);

  // Ensure the tag is added -- message size shouldn't include
  // BigStruct.
  EXPECT_EQ((GenSize<Var1, Var1>(0x1000, {"x"}, {"."})(&F)), 9);
}

TEST_F(PJTest, VariantMoveCaseNoTag) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var4 T{.value = {.x = -1}, .tag = Var4::Kind::undef};

  Transcode<Var1, Var4>(&F, &T, {"x"});

  EXPECT_EQ(T.tag, Var4::Kind::x);
  EXPECT_EQ(T.value.x, 42);
}

TEST_F(PJTest, VariantMoveCase) {
  Var1 F{.value = {.x = 42}, .tag = Var1::Kind::x};
  Var4 T{.value = {.x = -1}, .tag = Var4::Kind::undef};

  OnMatch<0, Var4>("x", [&](const Var4& T) {
    EXPECT_EQ(T.tag, Var4::Kind::x);
    EXPECT_EQ(T.value.x, 42);
  });

  Transcode<Var1, Var4>(&F, &T, {"x"}, {"."}, {"."});
}

TEST_F(PJTest, VariantMoveCase2) {
  Var3 F{.value = {.x = 42}, .tag = Var3::Kind::x};
  Var4 T{.value = {.x = -1}, .tag = Var4::Kind::undef};

  OnMatch<0, Var4>("x", [&](const Var4& T) {
    EXPECT_EQ(T.tag, Var4::Kind::x);
    EXPECT_EQ(T.value.x, 42);
  });

  Transcode<Var3, Var4>(&F, &T, {"x"}, {"."}, {"."});

  EXPECT_EQ((GenSize<Var3, Var4>(0x1000, {"x"}, {"."})(&F)), 9);
}

TEST_F(PJTest, VariantAddTagField) {
  Outer2 F{.z = 0xab};
  Outer T{.v = {.tag = Var4::Kind::undef}, .z = 0};

  Transcode<Outer2, Outer, Outer>(&F, &T, {}, {"v", "."}, {});

  EXPECT_EQ(T.v.tag, Var4::Kind::undef);
  EXPECT_EQ(T.z, 0xab);
}

// TODO: try this without path, when supported
TEST_F(PJTest, VariantRemoveTagField) {
  Outer F{.v = {.tag = Var4::Kind::undef}, .z = 0xab};
  Outer2 T{.z = 0};

  Transcode<Outer, Outer2, Outer>(&F, &T, {"v", "undef"}, {"v", "."}, {});

  EXPECT_EQ(T.z, 0xab);
}

TEST_F(PJTest, VariantSameNestedPathNoTag) {
  Outer F{.v = Var4{.value = {.x = 42}, .tag = Var4::Kind::x}};
  Outer T{.v = Var4{.value = {.x = -1}, .tag = Var4::Kind::undef}};

  Transcode<Outer>(&F, &T, {"v", "x"});

  EXPECT_EQ(F.v.tag, Var4::Kind::x);
  EXPECT_EQ(F.v.value.x, 42);
}

// TODO: Enable this test when SizeTarget supports variants.
#if 0
TEST_F(PJTest, VariantSameNoPathNestedTag) {
  Outer F{.v = Var4{.value = {.x = 42}, .tag = Var4::Kind::x}};
  Outer T{.v = Var4{.value = {.x = -1}, .tag = Var4::Kind::undef}};

  OnMatch<0, Outer>("x", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::x);
    EXPECT_EQ(T.v.value.x, 42);
  });

  Transcode<Outer>(&F, &T, {}, {"v", "."}, {"v", "."});
}
#endif

TEST_F(PJTest, VariantSameNestedPathNestedTag) {
  Outer F{.v = Var4{.value = {.x = 42}, .tag = Var4::Kind::x}};
  Outer T{.v = Var4{.value = {.x = -1}, .tag = Var4::Kind::undef}};

  OnMatch<0, Outer>("x", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::x);
    EXPECT_EQ(T.v.value.x, 42);
  });

  Transcode<Outer>(&F, &T, {"v", "x"}, {"v", "."}, {"v", "."});
}

TEST_F(PJTest, VariantSameNestedPathNestedTagAddFieldAfter) {
  Outer F{.v = Var4{.value = {.x = 42}, .tag = Var4::Kind::x}, .z = 0xab};
  Outer T{.v = Var4{.value = {.x = -1}, .tag = Var4::Kind::undef}, .z = 0};

  OnMatch<0, Outer>("x", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::x);
    EXPECT_EQ(T.v.value.x, 42);
    EXPECT_EQ(T.z, 0xab);
  });

  Transcode<Outer>(&F, &T, {"v", "x"}, {"v", "."}, {"v", "."});
}

TEST_F(PJTest, VariantNestedTagSize) {
  BigOuter F{.v = Var3{.value = {.x = 42}, .tag = Var3::Kind::x}};
  EXPECT_EQ(GenSize<BigOuter>(0x400, {"v", "x"}, {"v", "."})(&F), 9);
}

TEST_F(PJTest, VariantDispatchDefault) {
  Outer2 F{.z = 0xab};
  Outer T{.v = {.tag = Var4::Kind::w}, .z = 0};

  OnMatch<0, Outer>("undef", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  Transcode<Outer2, Outer, Outer>(&F, &T, {}, {"v", "."}, {"v", "."});
}

// TODO: Try without path, when supported.
TEST_F(PJTest, VariantDispatchUndef) {
  Outer F{.v = {.tag = Var4::Kind::undef}, .z = 0xab};
  Outer T{.v = {.value = {.w = 0}, .tag = Var4::Kind::w}, .z = 0};

  OnMatch<0, Outer>("undef", [&](const Outer& T) {
    EXPECT_EQ(T.v.tag, Var4::Kind::undef);
    EXPECT_EQ(T.z, 0xab);
  });

  Transcode<Outer>(&F, &T, {"v", "undef"}, {"v", "."}, {"v", "."});
}

// TODO: try without path
TEST_F(PJTest, VariantDifferentDispatchTag) {
  Outer3 F{
      .a = {.value = {.w = 0x11}, .tag = Var4::Kind::w},
      .b = {.value = {.x = 0x22222222}, .tag = Var4::Kind::x},
  };
  Outer3 T{
      .a = {.tag = Var4::Kind::undef},
      .b = {.tag = Var4::Kind::undef},
  };

  OnMatch<0, Outer3>("x", [&T](const Outer3& T2) {
    EXPECT_EQ(&T, &T2);
    EXPECT_EQ(T.a.tag, Var4::Kind::w);
    EXPECT_EQ(T.a.value.w, T.a.value.w);
    EXPECT_EQ(T.b.tag, Var4::Kind::x);
    EXPECT_EQ(T.b.value.x, T.b.value.x);
  });

  Transcode<Outer3>(&F, &T, {"a", "w"}, {"a", "."}, {"b", "."});

  EXPECT_EQ(GenSize<Outer3>(0x400, {"a", "w"}, {"a", "."})(&F), 11);
}

}  // namespace pj
