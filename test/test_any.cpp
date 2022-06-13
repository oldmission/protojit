#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "harness.hpp"
#include "pj/any.hpp"
#include "pj/protojit.hpp"
#include "test/any.pj.hpp"

namespace pj {

TEST_F(PJTest, IntStructTest) {
  Int32 x{.i = 42};
  Any y;

  auto results = transcode(
      Options<Int32, Any>{.from = &x, .to = &y, .expect_dec_buffer = true});

  EXPECT_EQ(y.kind(), Any::Kind::Struct);

  auto str = AnyStruct(y);
  EXPECT_EQ(str.numFields(), 1);

  EXPECT_EQ(str.name().size(), 1);
  EXPECT_EQ((std::string_view{str.name()[0].base(), str.name()[0].size()}),
            "Int32");

  auto field = str.getField(0);
  EXPECT_EQ(field.name(), "i");
  EXPECT_EQ(field.value().kind(), Any::Kind::Int);
  EXPECT_EQ(AnyInt(field.value()).getValue<uint64_t>(), 42);
}

TEST_F(PJTest, WrappedAnyTest) {
  WrapperI src{.x = 3, .i = {.i = 22}};
  WrapperA dst;

  auto results = transcode(Options<WrapperI, WrapperA>{
      .from = &src, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.x, 3);

  auto y = dst.i;
  EXPECT_EQ(y.kind(), Any::Kind::Struct);

  auto str = AnyStruct(y);
  EXPECT_EQ(str.numFields(), 1);

  auto field = str.getField(0);
  EXPECT_EQ(field.name(), "i");
  EXPECT_EQ(field.value().kind(), Any::Kind::Int);
  EXPECT_EQ(AnyInt(field.value()).getValue<uint64_t>(), 22);
}

TEST_F(PJTest, VariantTest) {
  Var1 src{Var1::x, 77};
  Any dst;

  auto results = transcode(
      Options<Var1, Any>{.from = &src, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.kind(), Any::Kind::Variant);
  EXPECT_EQ(AnyVariant(dst).termName(), "x");

  auto term = AnyVariant(dst).term();
  EXPECT_EQ(term.kind(), Any::Kind::Int);
  EXPECT_EQ(AnyInt(term).getValue<int>(), 77);
}

TEST_F(PJTest, AnyInVariantTest) {
  Var1 src{Var1::y, 0xffffffffaaaaaaaaUL};
  Var2 dst;

  auto results = transcode(
      Options<Var1, Var2>{.from = &src, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.tag, Var2::Kind::y);
  EXPECT_EQ(dst.value.y.kind(), Any::Kind::Int);

  AnyInt i(dst.value.y);
  EXPECT_EQ(i.getValue<uint64_t>(), 0xffffffffaaaaaaaaUL);
}

TEST_F(PJTest, VectorOfInts) {
  int32_t ary[3] = {10, 20, 30};
  span<pj_int32> array{ary, 3};
  Any dst;

  auto results = transcode(Options<decltype(array), Any>{
      .from = &array, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.kind(), Any::Kind::Sequence);
  auto seq = AnySequence(dst);
  EXPECT_EQ(seq.size(), 3);
  EXPECT_EQ(seq[0].kind(), Any::Kind::Int);
  EXPECT_EQ(AnyInt(seq[0]).getValue<int>(), 10);
  EXPECT_EQ(AnyInt(seq[1]).getValue<int>(), 20);
  EXPECT_EQ(AnyInt(seq[2]).getValue<int>(), 30);
}

TEST_F(PJTest, VectorOfStructs) {
  Int32 ary[3] = {{.i = 10}, {.i = 20}, {.i = 30}};
  span<Int32> array{ary, 3};
  Any dst;

  auto results = transcode(Options<decltype(array), Any>{
      .from = &array, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.kind(), Any::Kind::Sequence);
  auto seq = AnySequence(dst);
  EXPECT_EQ(seq.size(), 3);
  EXPECT_EQ(seq[0].kind(), Any::Kind::Struct);
  EXPECT_EQ(AnyInt(AnyStruct(seq[0]).getField(0).value()).getValue<int>(), 10);
  EXPECT_EQ(AnyInt(AnyStruct(seq[1]).getField(0).value()).getValue<int>(), 20);
  EXPECT_EQ(AnyInt(AnyStruct(seq[2]).getField(0).value()).getValue<int>(), 30);
}

TEST_F(PJTest, ArrayOfStructs) {
  pj::array<Int32, 3> ary = {{Int32{.i = 10}, Int32{.i = 20}, Int32{.i = 30}}};
  Any dst;

  auto results = transcode(Options<decltype(ary), Any>{
      .from = &ary, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.kind(), Any::Kind::Sequence);
  auto seq = AnySequence(dst);
  EXPECT_EQ(seq.size(), 3);
  EXPECT_EQ(seq[0].kind(), Any::Kind::Struct);
  EXPECT_EQ(AnyInt(AnyStruct(seq[0]).getField(0).value()).getValue<int>(), 10);
  EXPECT_EQ(AnyInt(AnyStruct(seq[1]).getField(0).value()).getValue<int>(), 20);
  EXPECT_EQ(AnyInt(AnyStruct(seq[2]).getField(0).value()).getValue<int>(), 30);
}

TEST_F(PJTest, Unit) {
  Var3 src{Var3::x};
  Any dst;

  auto results = transcode(
      Options<Var3, Any>{.from = &src, .to = &dst, .expect_dec_buffer = true});

  EXPECT_EQ(dst.kind(), Any::Kind::Variant);
  EXPECT_EQ(AnyVariant(dst).termName(), "x");
  EXPECT_EQ(AnyVariant(dst).term().kind(), Any::Kind::Unit);
}

}  // namespace pj
