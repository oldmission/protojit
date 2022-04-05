#include <gtest/gtest.h>

#include "pj/context.hpp"
#include "pj/types.hpp"

namespace pj {

struct TmpIR2Test : public ::testing::Test {
  ProtoJitContext ctx;
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

TEST_F(TmpIR2Test, BasicArrayTest) {
  auto int_m_ty =
      types::IntType::get(&ctx.ctx_, types::Int{
                                         .width = Bytes(8),
                                         .alignment = Bytes(8),
                                         .sign = types::Int::Sign::kSigned,
                                     });
  auto int_p_ty =
      types::IntType::get(&ctx.ctx_, types::Int{
                                         .width = Bytes(8),
                                         .alignment = Bytes(1),
                                         .sign = types::Int::Sign::kSigned,
                                     });

  auto ary_m_ty = types::ArrayType::get(  //
      &ctx.ctx_,                          //
      types::Array{
          .elem = int_m_ty,
          .length = 4,
          .elem_size = Bytes(8),
          .alignment = Bytes(8),
      });

  auto ary_p_ty = types::ArrayType::get(  //
      &ctx.ctx_,                          //
      types::Array{
          .elem = int_p_ty,
          .length = 2,
          .elem_size = Bytes(8),
          .alignment = Bytes(1),
      });

  auto proto = types::ProtocolType::get(&ctx.ctx_, types::Protocol{
                                                       .head = ary_p_ty,
                                                   });

  ctx.addDecodeFunction("decode", proto, ary_m_ty, {});

  ctx.compile(/*new_pipeline=*/true);

  EXPECT_EQ(0, 0);
}
#endif

}  // namespace pj
