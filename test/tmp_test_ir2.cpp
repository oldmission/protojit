#include <gtest/gtest.h>

#include "pj/context.hpp"
#include "pj/types.hpp"

namespace pj {

struct TmpIR2Test : public ::testing::Test {
  ProtoJitContext ctx_;
};

#if 0
TEST_F(TmpIR2Test, BasicStructTest) {
  auto int_m_ty =
      types::IntType::get(&ctx_.ctx_, types::Int{
                                          .width = Bytes(8),
                                          .alignment = Bytes(8),
                                          .sign = types::Int::Sign::kSigned,
                                      });
  auto int_p_ty =
      types::IntType::get(&ctx_.ctx_, types::Int{
                                          .width = Bytes(8),
                                          .alignment = Bytes(1),
                                          .sign = types::Int::Sign::kSigned,
                                      });

  llvm::SmallVector<types::StructField, 2> fields;
  fields.push_back(types::StructField{
      .type = int_m_ty,
      .name = "fld",
      .offset = Bytes(0),
  });

  auto struct_m_ty =
      types::StructType::get(&ctx_.ctx_, types::TypeDomain::kHost,
                             llvm::ArrayRef<llvm::StringRef>{"thing"});
  struct_m_ty.setTypeData({
      .fields = fields,
      .size = Bytes(8),
      .alignment = Bytes(8),
  });

  fields.clear();
  fields.push_back(types::StructField{
      .type = int_p_ty,
      .name = "fld",
      .offset = Bytes(0),
  });
  auto struct_p_ty =
      types::StructType::get(&ctx_.ctx_, types::TypeDomain::kWire,
                             llvm::ArrayRef<llvm::StringRef>{"thing"});
  struct_p_ty.setTypeData({
      .fields = fields,
      .size = Bytes(8),
      .alignment = Bytes(1),
  });

  auto proto = types::ProtocolType::get(&ctx_.ctx_,
                                        types::Protocol{.head = struct_p_ty});

  // ctx_.addEncodeFunction("encode", struct_m_ty, {}, proto);
  ctx_.addDecodeFunction("decode", proto, struct_m_ty, {});

  ctx_.compile(/*new_pipeline=*/true);

  EXPECT_EQ(0, 0);
}
#else

TEST_F(TmpIR2Test, BasicVariantTest) {
  auto int_m_ty =
      types::IntType::get(&ctx_.ctx_, types::Int{
                                          .width = Bytes(8),
                                          .alignment = Bytes(8),
                                          .sign = types::Int::Sign::kSigned,
                                      });
  auto int_p_ty =
      types::IntType::get(&ctx_.ctx_, types::Int{
                                          .width = Bytes(8),
                                          .alignment = Bytes(1),
                                          .sign = types::Int::Sign::kSigned,
                                      });

  llvm::SmallVector<types::Term, 1> terms;
  terms.push_back(types::Term{
      .name = "term",
      .type = int_m_ty,
      .tag = 1,
  });

  auto var_m_ty = types::InlineVariantType::get(
      &ctx_.ctx_, types::TypeDomain::kHost, types::Name{"thing"});
  var_m_ty.setTypeData(types::InlineVariant{
      .terms = terms,
      .term_offset = Bytes(0),
      .term_size = Bytes(8),
      .tag_offset = Bytes(8),
      .tag_width = Bytes(1),
      .size = Bytes(16),
      .alignment = Bytes(8),
  });

  terms.clear();
  terms.push_back(types::Term{
      .name = "term",
      .type = int_p_ty,
      .tag = 1,
  });

  auto var_p_ty = types::OutlineVariantType::get(
      &ctx_.ctx_, types::TypeDomain::kWire, types::Name{"thing"});

  var_p_ty.setTypeData(types::OutlineVariant{
      .terms = terms,
      .tag_width = Bytes(1),
      .tag_alignment = Bytes(1),
      .term_offset = Bytes(1),
  });

  auto proto = types::ProtocolType::get(&ctx_.ctx_, types::Protocol{
                                                        .head = var_p_ty,
                                                    });

  ctx_.addDecodeFunction("decode", proto, var_m_ty,
                         {
                             {"undef", reinterpret_cast<void*>(0x2345)},
                             {"term", reinterpret_cast<void*>(0x1234)},
                         });

  ctx_.compile(/*new_pipeline=*/true);

  EXPECT_EQ(0, 0);
}
#endif

}  // namespace pj
