#pragma once
#include <cstddef>
#include <string_view>
#include "pj/runtime.h"
#include "pj/traits.hpp"

namespace v0_1 {
namespace pj {
struct Width;
}
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Array;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
enum class FloatWidth : unsigned char;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Float;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Undef;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
enum class VectorSplitType : unsigned char;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct VectorSplit;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct TermAttribute;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Term;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct InlineVariant;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
enum class Sign : unsigned char;
}
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Int;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct OutlineVariant;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct StructField;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Struct;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Unit;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
enum class ReferenceMode : unsigned char;
}
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Vector;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Type;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Protocol;
}
}  // namespace pj
}  // namespace v0_1

namespace v0_1 {
namespace pj {
namespace reflect {
struct Schema;
}
}  // namespace pj
}  // namespace v0_1

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::Width> {
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
struct BuildPJType<::v0_1::pj::reflect::Array> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _3 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                     /*sign=*/PJ_SIGN_SIGNED);
    fields[0] = PJCreateStructField(/*name=*/"elem", /*type=*/_3, /*offset=*/0);
    const auto* _4 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                     /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"length", /*type=*/_4, /*offset=*/32);
    const auto* _5 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"elem_size", /*type=*/_5, /*offset=*/96);
    const auto* _6 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
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
struct BuildPJType<::v0_1::pj::reflect::FloatWidth> {
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
struct BuildPJType<::v0_1::pj::reflect::Float> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[2];
    const auto* _14 =
        BuildPJType<::v0_1::pj::reflect::FloatWidth>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_14, /*offset=*/0);
    const auto* _15 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
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
struct BuildPJType<::v0_1::pj::reflect::Undef> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[1];
    const auto* _18 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"is_default", /*type=*/_18, /*offset=*/0);
    const char* _19[3] = {"pj", "reflect", "Undef"};
    const PJStructType* _20 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_19, /*type_domain=*/domain,
        /*num_fields=*/1, /*fields=*/fields, /*size=*/8, /*alignment=*/8);
    return _20;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::VectorSplitType> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _21 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kInline", /*type=*/_21, /*tag=*/1);
    const PJUnitType* _22 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kOutline", /*type=*/_22, /*tag=*/2);
    const PJUnitType* _23 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_23, /*tag=*/0);
    const char* _24[3] = {"pj", "reflect", "VectorSplitType"};
    const PJInlineVariantType* _25 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_24, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _25;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::VectorSplit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _26 =
        BuildPJType<::v0_1::pj::reflect::VectorSplitType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_26, /*offset=*/0);
    const auto* _27 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] = PJCreateStructField(/*name=*/"inline_length", /*type=*/_27,
                                    /*offset=*/8);
    const auto* _30 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _29 = PJCreateVectorType(
        ctx, /*elem=*/_30, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _28 = PJCreateVectorType(
        ctx, /*elem=*/_29, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_28, /*offset=*/72);
    const auto* _31 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_31,
                                    /*offset=*/200);
    const char* _32[3] = {"pj", "reflect", "VectorSplit"};
    const PJStructType* _33 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_32, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _33;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::TermAttribute> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[2];
    const auto* _34 =
        BuildPJType<::v0_1::pj::reflect::Undef>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"undef", /*type=*/_34, /*tag=*/0);
    const auto* _35 =
        BuildPJType<::v0_1::pj::reflect::VectorSplit>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"vector_split", /*type=*/_35, /*tag=*/1);
    const char* _36[3] = {"pj", "reflect", "TermAttribute"};
    const PJInlineVariantType* _37 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_36, /*type_domain=*/domain,
        /*num_terms=*/2, /*terms=*/terms, /*default_term=*/0, /*term_offset=*/0,
        /*term_size=*/208, /*tag_offset=*/208, /*tag_width=*/8, /*size=*/216,
        /*alignment=*/8);
    return _37;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Term> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _39 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _38 = PJCreateVectorType(
        ctx, /*elem=*/_39, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_38, /*offset=*/0);
    const auto* _40 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"type", /*type=*/_40, /*offset=*/128);
    const auto* _41 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"tag", /*type=*/_41, /*offset=*/160);
    const auto* _43 =
        BuildPJType<::v0_1::pj::reflect::TermAttribute>::build(ctx, domain);
    const auto* _42 = PJCreateVectorType(
        ctx, /*elem=*/_43, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] = PJCreateStructField(/*name=*/"attributes", /*type=*/_42,
                                    /*offset=*/224);
    const char* _44[3] = {"pj", "reflect", "Term"};
    const PJStructType* _45 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_44, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/352, /*alignment=*/8);
    return _45;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::InlineVariant> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[9];
    const auto* _48 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _47 = PJCreateVectorType(
        ctx, /*elem=*/_48, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _46 = PJCreateVectorType(
        ctx, /*elem=*/_47, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_46, /*offset=*/0);
    const auto* _50 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _49 = PJCreateVectorType(
        ctx, /*elem=*/_50, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_49, /*offset=*/128);
    const auto* _51 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_51,
                                    /*offset=*/256);
    const auto* _52 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_52,
                                    /*offset=*/320);
    const auto* _53 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] =
        PJCreateStructField(/*name=*/"term_size", /*type=*/_53, /*offset=*/384);
    const auto* _54 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"tag_offset", /*type=*/_54,
                                    /*offset=*/448);
    const auto* _55 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_55, /*offset=*/512);
    const auto* _56 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"size", /*type=*/_56, /*offset=*/576);
    const auto* _57 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_57, /*offset=*/640);
    const char* _58[3] = {"pj", "reflect", "InlineVariant"};
    const PJStructType* _59 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_58, /*type_domain=*/domain,
        /*num_fields=*/9, /*fields=*/fields, /*size=*/704, /*alignment=*/8);
    return _59;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::Sign> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[4];
    const PJUnitType* _60 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kSigned", /*type=*/_60, /*tag=*/1);
    const PJUnitType* _61 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kSignless", /*type=*/_61, /*tag=*/3);
    const PJUnitType* _62 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"kUnsigned", /*type=*/_62, /*tag=*/2);
    const PJUnitType* _63 = PJCreateUnitType(ctx);
    terms[3] = PJCreateTerm(/*name=*/"undef", /*type=*/_63, /*tag=*/0);
    const char* _64[2] = {"pj", "Sign"};
    const PJInlineVariantType* _65 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_64, /*type_domain=*/domain,
        /*num_terms=*/4, /*terms=*/terms, /*default_term=*/3, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _65;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Int> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[3];
    const auto* _66 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_66, /*offset=*/0);
    const auto* _67 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_67, /*offset=*/64);
    const auto* _68 = BuildPJType<::v0_1::pj::Sign>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"sign", /*type=*/_68, /*offset=*/128);
    const char* _69[3] = {"pj", "reflect", "Int"};
    const PJStructType* _70 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_69, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/136, /*alignment=*/8);
    return _70;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::OutlineVariant> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[7];
    const auto* _73 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _72 = PJCreateVectorType(
        ctx, /*elem=*/_73, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _71 = PJCreateVectorType(
        ctx, /*elem=*/_72, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_71, /*offset=*/0);
    const auto* _75 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _74 = PJCreateVectorType(
        ctx, /*elem=*/_75, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_74, /*offset=*/128);
    const auto* _76 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_76,
                                    /*offset=*/256);
    const auto* _77 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_77, /*offset=*/320);
    const auto* _78 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"tag_alignment", /*type=*/_78,
                                    /*offset=*/384);
    const auto* _79 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_79,
                                    /*offset=*/448);
    const auto* _80 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"term_alignment", /*type=*/_80,
                                    /*offset=*/512);
    const char* _81[3] = {"pj", "reflect", "OutlineVariant"};
    const PJStructType* _82 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_81, /*type_domain=*/domain,
        /*num_fields=*/7, /*fields=*/fields, /*size=*/576, /*alignment=*/8);
    return _82;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::StructField> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[3];
    const auto* _83 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_83, /*offset=*/0);
    const auto* _85 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _84 = PJCreateVectorType(
        ctx, /*elem=*/_85, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"name", /*type=*/_84, /*offset=*/32);
    const auto* _86 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"offset", /*type=*/_86, /*offset=*/160);
    const char* _87[3] = {"pj", "reflect", "StructField"};
    const PJStructType* _88 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_87, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _88;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Struct> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _91 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _90 = PJCreateVectorType(
        ctx, /*elem=*/_91, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _89 = PJCreateVectorType(
        ctx, /*elem=*/_90, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_89, /*offset=*/0);
    const auto* _93 =
        BuildPJType<::v0_1::pj::reflect::StructField>::build(ctx, domain);
    const auto* _92 = PJCreateVectorType(
        ctx, /*elem=*/_93, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"fields", /*type=*/_92, /*offset=*/128);
    const auto* _94 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"size", /*type=*/_94, /*offset=*/256);
    const auto* _95 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_95, /*offset=*/320);
    const char* _96[3] = {"pj", "reflect", "Struct"};
    const PJStructType* _97 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_96, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/384, /*alignment=*/8);
    return _97;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Unit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[0];
    const char* _98[3] = {"pj", "reflect", "Unit"};
    const PJStructType* _99 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_98, /*type_domain=*/domain,
        /*num_fields=*/0, /*fields=*/fields, /*size=*/0, /*alignment=*/8);
    return _99;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::ReferenceMode> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _100 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOffset", /*type=*/_100, /*tag=*/2);
    const PJUnitType* _101 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kPointer", /*type=*/_101, /*tag=*/1);
    const PJUnitType* _102 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_102, /*tag=*/0);
    const char* _103[2] = {"pj", "ReferenceMode"};
    const PJInlineVariantType* _104 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_103, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _104;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Vector> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[14];
    const auto* _105 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"elem", /*type=*/_105, /*offset=*/0);
    const auto* _106 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] = PJCreateStructField(/*name=*/"elem_width", /*type=*/_106,
                                    /*offset=*/32);
    const auto* _107 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"min_length", /*type=*/_107,
                                    /*offset=*/96);
    const auto* _108 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[3] = PJCreateStructField(/*name=*/"max_length", /*type=*/_108,
                                    /*offset=*/160);
    const auto* _109 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[4] = PJCreateStructField(/*name=*/"ppl_count", /*type=*/_109,
                                    /*offset=*/224);
    const auto* _110 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"length_offset", /*type=*/_110,
                                    /*offset=*/288);
    const auto* _111 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"length_size", /*type=*/_111,
                                    /*offset=*/352);
    const auto* _112 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] = PJCreateStructField(/*name=*/"ref_offset", /*type=*/_112,
                                    /*offset=*/416);
    const auto* _113 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"ref_size", /*type=*/_113, /*offset=*/480);
    const auto* _114 =
        BuildPJType<::v0_1::pj::ReferenceMode>::build(ctx, domain);
    fields[9] = PJCreateStructField(/*name=*/"reference_mode", /*type=*/_114,
                                    /*offset=*/544);
    const auto* _115 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[10] = PJCreateStructField(/*name=*/"inline_payload_offset",
                                     /*type=*/_115, /*offset=*/552);
    const auto* _116 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[11] = PJCreateStructField(/*name=*/"partial_payload_offset",
                                     /*type=*/_116, /*offset=*/616);
    const auto* _117 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[12] =
        PJCreateStructField(/*name=*/"size", /*type=*/_117, /*offset=*/680);
    const auto* _118 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[13] = PJCreateStructField(/*name=*/"alignment", /*type=*/_118,
                                     /*offset=*/744);
    const char* _119[3] = {"pj", "reflect", "Vector"};
    const PJStructType* _120 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_119, /*type_domain=*/domain,
        /*num_fields=*/14, /*fields=*/fields, /*size=*/808, /*alignment=*/8);
    return _120;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Type> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[9];
    const auto* _121 =
        BuildPJType<::v0_1::pj::reflect::Array>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"Array", /*type=*/_121, /*tag=*/7);
    const auto* _122 =
        BuildPJType<::v0_1::pj::reflect::Float>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"Float", /*type=*/_122, /*tag=*/2);
    const auto* _123 =
        BuildPJType<::v0_1::pj::reflect::InlineVariant>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"InlineVariant", /*type=*/_123, /*tag=*/5);
    const auto* _124 =
        BuildPJType<::v0_1::pj::reflect::Int>::build(ctx, domain);
    terms[3] = PJCreateTerm(/*name=*/"Int", /*type=*/_124, /*tag=*/1);
    const auto* _125 =
        BuildPJType<::v0_1::pj::reflect::OutlineVariant>::build(ctx, domain);
    terms[4] =
        PJCreateTerm(/*name=*/"OutlineVariant", /*type=*/_125, /*tag=*/6);
    const auto* _126 =
        BuildPJType<::v0_1::pj::reflect::Struct>::build(ctx, domain);
    terms[5] = PJCreateTerm(/*name=*/"Struct", /*type=*/_126, /*tag=*/4);
    const auto* _127 =
        BuildPJType<::v0_1::pj::reflect::Unit>::build(ctx, domain);
    terms[6] = PJCreateTerm(/*name=*/"Unit", /*type=*/_127, /*tag=*/3);
    const auto* _128 =
        BuildPJType<::v0_1::pj::reflect::Vector>::build(ctx, domain);
    terms[7] = PJCreateTerm(/*name=*/"Vector", /*type=*/_128, /*tag=*/8);
    const PJUnitType* _129 = PJCreateUnitType(ctx);
    terms[8] = PJCreateTerm(/*name=*/"undef", /*type=*/_129, /*tag=*/0);
    const char* _130[3] = {"pj", "reflect", "Type"};
    const PJInlineVariantType* _131 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_130, /*type_domain=*/domain,
        /*num_terms=*/9, /*terms=*/terms, /*default_term=*/8, /*term_offset=*/0,
        /*term_size=*/808, /*tag_offset=*/808, /*tag_width=*/8, /*size=*/816,
        /*alignment=*/8);
    return _131;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Protocol> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _132 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"pj_version", /*type=*/_132, /*offset=*/0);
    const auto* _133 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"head", /*type=*/_133, /*offset=*/32);
    const auto* _134 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"buffer_offset", /*type=*/_134,
                                    /*offset=*/64);
    const auto* _136 =
        BuildPJType<::v0_1::pj::reflect::Type>::build(ctx, domain);
    const auto* _135 = PJCreateVectorType(
        ctx, /*elem=*/_136, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] =
        PJCreateStructField(/*name=*/"types", /*type=*/_135, /*offset=*/128);
    const char* _137[3] = {"pj", "reflect", "Protocol"};
    const PJStructType* _138 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_137, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/256, /*alignment=*/8);
    return _138;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJProtocol<::v0_1::pj::reflect::Schema> {
  using Head = ::v0_1::pj::reflect::Protocol;
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const auto* _139 =
        BuildPJType<::v0_1::pj::reflect::Protocol>::build(ctx, domain);
    return PJCreateProtocolType(ctx, _139, 0);
  }
};
}  // namespace gen

}  // namespace pj
