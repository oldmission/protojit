#pragma once
#include <cstddef>
#include <string_view>
#include "pj/runtime.h"
#include "pj/traits.hpp"

namespace v0_2 {
namespace pj {
struct Width;
}
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Array;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
enum class FloatWidth : unsigned char;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Float;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
enum class ShortIntType : unsigned char;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct ShortInt;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Undef;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
enum class VectorSplitType : unsigned char;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct VectorSplit;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct TermAttribute;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Term;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct InlineVariant;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
enum class Sign : unsigned char;
}
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Int;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct OutlineVariant;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct StructField;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Struct;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Unit;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
enum class ReferenceMode : unsigned char;
}
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Vector;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Type;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Protocol;
}
}  // namespace pj
}  // namespace v0_2

namespace v0_2 {
namespace pj {
namespace reflect {
struct Schema;
}
}  // namespace pj
}  // namespace v0_2

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::Width> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[1];
    const auto* _0 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                     /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"bits_", /*type=*/_0, /*offset=*/0);
    const char* _1[2] = {"pj", "Width"};
    const PJStructType* _2 = PJCreateStructType(
        ctx, /*name_size=*/2, /*name=*/_1, /*type_domain=*/domain,
        /*num_fields=*/1, /*fields=*/fields, /*size=*/64, /*alignment=*/8);
    return _2;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Array> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _3 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                     /*sign=*/PJ_SIGN_SIGNED);
    fields[0] = PJCreateStructField(/*name=*/"elem", /*type=*/_3, /*offset=*/0);
    const auto* _4 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                     /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"length", /*type=*/_4, /*offset=*/32);
    const auto* _5 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"elem_size", /*type=*/_5, /*offset=*/96);
    const auto* _6 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_6, /*offset=*/160);
    const char* _7[3] = {"pj", "reflect", "Array"};
    const PJStructType* _8 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_7, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _8;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::FloatWidth> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _9 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"k32", /*type=*/_9, /*tag=*/1);
    const PJUnitType* _10 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"k64", /*type=*/_10, /*tag=*/2);
    const PJUnitType* _11 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_11, /*tag=*/0);
    const char* _12[3] = {"pj", "reflect", "FloatWidth"};
    const PJInlineVariantType* _13 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_12, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _13;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Float> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[2];
    const auto* _14 =
        BuildPJType<::v0_2::pj::reflect::FloatWidth>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_14, /*offset=*/0);
    const auto* _15 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_15, /*offset=*/8);
    const char* _16[3] = {"pj", "reflect", "Float"};
    const PJStructType* _17 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_16, /*type_domain=*/domain,
        /*num_fields=*/2, /*fields=*/fields, /*size=*/72, /*alignment=*/8);
    return _17;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::ShortIntType> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _18 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOriginal", /*type=*/_18, /*tag=*/2);
    const PJUnitType* _19 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kShort", /*type=*/_19, /*tag=*/1);
    const PJUnitType* _20 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_20, /*tag=*/0);
    const char* _21[3] = {"pj", "reflect", "ShortIntType"};
    const PJInlineVariantType* _22 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_21, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _22;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::ShortInt> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _23 =
        BuildPJType<::v0_2::pj::reflect::ShortIntType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_23, /*offset=*/0);
    const auto* _24 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"threshold", /*type=*/_24, /*offset=*/8);
    const auto* _27 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _26 = PJCreateVectorType(
        ctx, /*elem=*/_27, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _25 = PJCreateVectorType(
        ctx, /*elem=*/_26, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_25, /*offset=*/72);
    const auto* _28 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_28,
                                    /*offset=*/200);
    const char* _29[3] = {"pj", "reflect", "ShortInt"};
    const PJStructType* _30 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_29, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _30;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Undef> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[1];
    const auto* _31 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"is_default", /*type=*/_31, /*offset=*/0);
    const char* _32[3] = {"pj", "reflect", "Undef"};
    const PJStructType* _33 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_32, /*type_domain=*/domain,
        /*num_fields=*/1, /*fields=*/fields, /*size=*/8, /*alignment=*/8);
    return _33;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::VectorSplitType> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _34 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kInline", /*type=*/_34, /*tag=*/1);
    const PJUnitType* _35 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kOutline", /*type=*/_35, /*tag=*/2);
    const PJUnitType* _36 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_36, /*tag=*/0);
    const char* _37[3] = {"pj", "reflect", "VectorSplitType"};
    const PJInlineVariantType* _38 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_37, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _38;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::VectorSplit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _39 =
        BuildPJType<::v0_2::pj::reflect::VectorSplitType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_39, /*offset=*/0);
    const auto* _40 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] = PJCreateStructField(/*name=*/"inline_length", /*type=*/_40,
                                    /*offset=*/8);
    const auto* _43 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _42 = PJCreateVectorType(
        ctx, /*elem=*/_43, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _41 = PJCreateVectorType(
        ctx, /*elem=*/_42, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_41, /*offset=*/72);
    const auto* _44 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_44,
                                    /*offset=*/200);
    const char* _45[3] = {"pj", "reflect", "VectorSplit"};
    const PJStructType* _46 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_45, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _46;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::TermAttribute> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const auto* _47 =
        BuildPJType<::v0_2::pj::reflect::ShortInt>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"short_int", /*type=*/_47, /*tag=*/2);
    const auto* _48 =
        BuildPJType<::v0_2::pj::reflect::Undef>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"undef", /*type=*/_48, /*tag=*/0);
    const auto* _49 =
        BuildPJType<::v0_2::pj::reflect::VectorSplit>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"vector_split", /*type=*/_49, /*tag=*/1);
    const char* _50[3] = {"pj", "reflect", "TermAttribute"};
    const PJInlineVariantType* _51 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_50, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/1, /*term_offset=*/0,
        /*term_size=*/208, /*tag_offset=*/208, /*tag_width=*/8, /*size=*/216,
        /*alignment=*/8);
    return _51;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Term> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _53 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _52 = PJCreateVectorType(
        ctx, /*elem=*/_53, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_52, /*offset=*/0);
    const auto* _54 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"type", /*type=*/_54, /*offset=*/128);
    const auto* _55 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"tag", /*type=*/_55, /*offset=*/160);
    const auto* _57 =
        BuildPJType<::v0_2::pj::reflect::TermAttribute>::build(ctx, domain);
    const auto* _56 = PJCreateVectorType(
        ctx, /*elem=*/_57, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] = PJCreateStructField(/*name=*/"attributes", /*type=*/_56,
                                    /*offset=*/224);
    const char* _58[3] = {"pj", "reflect", "Term"};
    const PJStructType* _59 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_58, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/352, /*alignment=*/8);
    return _59;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::InlineVariant> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[9];
    const auto* _62 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _61 = PJCreateVectorType(
        ctx, /*elem=*/_62, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _60 = PJCreateVectorType(
        ctx, /*elem=*/_61, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_60, /*offset=*/0);
    const auto* _64 =
        BuildPJType<::v0_2::pj::reflect::Term>::build(ctx, domain);
    const auto* _63 = PJCreateVectorType(
        ctx, /*elem=*/_64, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_63, /*offset=*/128);
    const auto* _65 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_65,
                                    /*offset=*/256);
    const auto* _66 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_66,
                                    /*offset=*/320);
    const auto* _67 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[4] =
        PJCreateStructField(/*name=*/"term_size", /*type=*/_67, /*offset=*/384);
    const auto* _68 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"tag_offset", /*type=*/_68,
                                    /*offset=*/448);
    const auto* _69 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[6] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_69, /*offset=*/512);
    const auto* _70 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"size", /*type=*/_70, /*offset=*/576);
    const auto* _71 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_71, /*offset=*/640);
    const char* _72[3] = {"pj", "reflect", "InlineVariant"};
    const PJStructType* _73 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_72, /*type_domain=*/domain,
        /*num_fields=*/9, /*fields=*/fields, /*size=*/704, /*alignment=*/8);
    return _73;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::Sign> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[4];
    const PJUnitType* _74 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kSigned", /*type=*/_74, /*tag=*/1);
    const PJUnitType* _75 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kSignless", /*type=*/_75, /*tag=*/3);
    const PJUnitType* _76 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"kUnsigned", /*type=*/_76, /*tag=*/2);
    const PJUnitType* _77 = PJCreateUnitType(ctx);
    terms[3] = PJCreateTerm(/*name=*/"undef", /*type=*/_77, /*tag=*/0);
    const char* _78[2] = {"pj", "Sign"};
    const PJInlineVariantType* _79 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_78, /*type_domain=*/domain,
        /*num_terms=*/4, /*terms=*/terms, /*default_term=*/3, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _79;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Int> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[3];
    const auto* _80 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_80, /*offset=*/0);
    const auto* _81 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_81, /*offset=*/64);
    const auto* _82 = BuildPJType<::v0_2::pj::Sign>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"sign", /*type=*/_82, /*offset=*/128);
    const char* _83[3] = {"pj", "reflect", "Int"};
    const PJStructType* _84 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_83, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/136, /*alignment=*/8);
    return _84;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::OutlineVariant> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[7];
    const auto* _87 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _86 = PJCreateVectorType(
        ctx, /*elem=*/_87, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _85 = PJCreateVectorType(
        ctx, /*elem=*/_86, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_85, /*offset=*/0);
    const auto* _89 =
        BuildPJType<::v0_2::pj::reflect::Term>::build(ctx, domain);
    const auto* _88 = PJCreateVectorType(
        ctx, /*elem=*/_89, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_88, /*offset=*/128);
    const auto* _90 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_90,
                                    /*offset=*/256);
    const auto* _91 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_91, /*offset=*/320);
    const auto* _92 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"tag_alignment", /*type=*/_92,
                                    /*offset=*/384);
    const auto* _93 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_93,
                                    /*offset=*/448);
    const auto* _94 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"term_alignment", /*type=*/_94,
                                    /*offset=*/512);
    const char* _95[3] = {"pj", "reflect", "OutlineVariant"};
    const PJStructType* _96 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_95, /*type_domain=*/domain,
        /*num_fields=*/7, /*fields=*/fields, /*size=*/576, /*alignment=*/8);
    return _96;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::StructField> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[3];
    const auto* _97 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_97, /*offset=*/0);
    const auto* _99 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _98 = PJCreateVectorType(
        ctx, /*elem=*/_99, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"name", /*type=*/_98, /*offset=*/32);
    const auto* _100 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"offset", /*type=*/_100, /*offset=*/160);
    const char* _101[3] = {"pj", "reflect", "StructField"};
    const PJStructType* _102 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_101, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _102;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Struct> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _105 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _104 = PJCreateVectorType(
        ctx, /*elem=*/_105, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _103 = PJCreateVectorType(
        ctx, /*elem=*/_104, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_103, /*offset=*/0);
    const auto* _107 =
        BuildPJType<::v0_2::pj::reflect::StructField>::build(ctx, domain);
    const auto* _106 = PJCreateVectorType(
        ctx, /*elem=*/_107, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"fields", /*type=*/_106, /*offset=*/128);
    const auto* _108 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"size", /*type=*/_108, /*offset=*/256);
    const auto* _109 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"alignment", /*type=*/_109,
                                    /*offset=*/320);
    const char* _110[3] = {"pj", "reflect", "Struct"};
    const PJStructType* _111 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_110, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/384, /*alignment=*/8);
    return _111;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Unit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[0];
    const char* _112[3] = {"pj", "reflect", "Unit"};
    const PJStructType* _113 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_112, /*type_domain=*/domain,
        /*num_fields=*/0, /*fields=*/fields, /*size=*/0, /*alignment=*/8);
    return _113;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::ReferenceMode> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _114 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOffset", /*type=*/_114, /*tag=*/2);
    const PJUnitType* _115 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kPointer", /*type=*/_115, /*tag=*/1);
    const PJUnitType* _116 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_116, /*tag=*/0);
    const char* _117[2] = {"pj", "ReferenceMode"};
    const PJInlineVariantType* _118 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_117, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _118;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Vector> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[14];
    const auto* _119 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"elem", /*type=*/_119, /*offset=*/0);
    const auto* _120 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[1] = PJCreateStructField(/*name=*/"elem_width", /*type=*/_120,
                                    /*offset=*/32);
    const auto* _121 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"min_length", /*type=*/_121,
                                    /*offset=*/96);
    const auto* _122 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[3] = PJCreateStructField(/*name=*/"max_length", /*type=*/_122,
                                    /*offset=*/160);
    const auto* _123 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[4] = PJCreateStructField(/*name=*/"ppl_count", /*type=*/_123,
                                    /*offset=*/224);
    const auto* _124 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"length_offset", /*type=*/_124,
                                    /*offset=*/288);
    const auto* _125 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"length_size", /*type=*/_125,
                                    /*offset=*/352);
    const auto* _126 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[7] = PJCreateStructField(/*name=*/"ref_offset", /*type=*/_126,
                                    /*offset=*/416);
    const auto* _127 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"ref_size", /*type=*/_127, /*offset=*/480);
    const auto* _128 =
        BuildPJType<::v0_2::pj::ReferenceMode>::build(ctx, domain);
    fields[9] = PJCreateStructField(/*name=*/"reference_mode", /*type=*/_128,
                                    /*offset=*/544);
    const auto* _129 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[10] = PJCreateStructField(/*name=*/"inline_payload_offset",
                                     /*type=*/_129, /*offset=*/552);
    const auto* _130 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[11] = PJCreateStructField(/*name=*/"partial_payload_offset",
                                     /*type=*/_130, /*offset=*/616);
    const auto* _131 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[12] =
        PJCreateStructField(/*name=*/"size", /*type=*/_131, /*offset=*/680);
    const auto* _132 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[13] = PJCreateStructField(/*name=*/"alignment", /*type=*/_132,
                                     /*offset=*/744);
    const char* _133[3] = {"pj", "reflect", "Vector"};
    const PJStructType* _134 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_133, /*type_domain=*/domain,
        /*num_fields=*/14, /*fields=*/fields, /*size=*/808, /*alignment=*/8);
    return _134;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Type> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[9];
    const auto* _135 =
        BuildPJType<::v0_2::pj::reflect::Array>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"Array", /*type=*/_135, /*tag=*/7);
    const auto* _136 =
        BuildPJType<::v0_2::pj::reflect::Float>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"Float", /*type=*/_136, /*tag=*/2);
    const auto* _137 =
        BuildPJType<::v0_2::pj::reflect::InlineVariant>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"InlineVariant", /*type=*/_137, /*tag=*/5);
    const auto* _138 =
        BuildPJType<::v0_2::pj::reflect::Int>::build(ctx, domain);
    terms[3] = PJCreateTerm(/*name=*/"Int", /*type=*/_138, /*tag=*/1);
    const auto* _139 =
        BuildPJType<::v0_2::pj::reflect::OutlineVariant>::build(ctx, domain);
    terms[4] =
        PJCreateTerm(/*name=*/"OutlineVariant", /*type=*/_139, /*tag=*/6);
    const auto* _140 =
        BuildPJType<::v0_2::pj::reflect::Struct>::build(ctx, domain);
    terms[5] = PJCreateTerm(/*name=*/"Struct", /*type=*/_140, /*tag=*/4);
    const auto* _141 =
        BuildPJType<::v0_2::pj::reflect::Unit>::build(ctx, domain);
    terms[6] = PJCreateTerm(/*name=*/"Unit", /*type=*/_141, /*tag=*/3);
    const auto* _142 =
        BuildPJType<::v0_2::pj::reflect::Vector>::build(ctx, domain);
    terms[7] = PJCreateTerm(/*name=*/"Vector", /*type=*/_142, /*tag=*/8);
    const PJUnitType* _143 = PJCreateUnitType(ctx);
    terms[8] = PJCreateTerm(/*name=*/"undef", /*type=*/_143, /*tag=*/0);
    const char* _144[3] = {"pj", "reflect", "Type"};
    const PJInlineVariantType* _145 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_144, /*type_domain=*/domain,
        /*num_terms=*/9, /*terms=*/terms, /*default_term=*/8, /*term_offset=*/0,
        /*term_size=*/808, /*tag_offset=*/808, /*tag_width=*/8, /*size=*/816,
        /*alignment=*/8);
    return _145;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Protocol> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _146 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"pj_version", /*type=*/_146, /*offset=*/0);
    const auto* _147 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"head", /*type=*/_147, /*offset=*/32);
    const auto* _148 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"buffer_offset", /*type=*/_148,
                                    /*offset=*/64);
    const auto* _150 =
        BuildPJType<::v0_2::pj::reflect::Type>::build(ctx, domain);
    const auto* _149 = PJCreateVectorType(
        ctx, /*elem=*/_150, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] =
        PJCreateStructField(/*name=*/"types", /*type=*/_149, /*offset=*/128);
    const char* _151[3] = {"pj", "reflect", "Protocol"};
    const PJStructType* _152 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_151, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/256, /*alignment=*/8);
    return _152;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJProtocol<::v0_2::pj::reflect::Schema> {
  using Head = ::v0_2::pj::reflect::Protocol;
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const auto* _153 =
        BuildPJType<::v0_2::pj::reflect::Protocol>::build(ctx, domain);
    return PJCreateProtocolType(ctx, _153, 0);
  }
};
}  // namespace gen

}  // namespace pj
