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
struct BuildPJType<::v0_1::pj::reflect::VectorSplitType> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _9 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kInline", /*type=*/_9, /*tag=*/1);
    const PJUnitType* _10 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kOutline", /*type=*/_10, /*tag=*/2);
    const PJUnitType* _11 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_11, /*tag=*/0);
    const char* _12[3] = {"pj", "reflect", "VectorSplitType"};
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
struct BuildPJType<::v0_1::pj::reflect::VectorSplit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _14 =
        BuildPJType<::v0_1::pj::reflect::VectorSplitType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_14, /*offset=*/0);
    const auto* _15 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] = PJCreateStructField(/*name=*/"inline_length", /*type=*/_15,
                                    /*offset=*/8);
    const auto* _18 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _17 = PJCreateVectorType(
        ctx, /*elem=*/_18, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _16 = PJCreateVectorType(
        ctx, /*elem=*/_17, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_16, /*offset=*/72);
    const auto* _19 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_19,
                                    /*offset=*/200);
    const char* _20[3] = {"pj", "reflect", "VectorSplit"};
    const PJStructType* _21 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_20, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _21;
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
    const PJUnitType* _22 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"undef", /*type=*/_22, /*tag=*/0);
    const auto* _23 =
        BuildPJType<::v0_1::pj::reflect::VectorSplit>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"vector_split", /*type=*/_23, /*tag=*/1);
    const char* _24[3] = {"pj", "reflect", "TermAttribute"};
    const PJInlineVariantType* _25 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_24, /*type_domain=*/domain,
        /*num_terms=*/2, /*terms=*/terms, /*default_term=*/0, /*term_offset=*/0,
        /*term_size=*/208, /*tag_offset=*/208, /*tag_width=*/8, /*size=*/216,
        /*alignment=*/8);
    return _25;
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
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_26, /*offset=*/0);
    const auto* _28 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"type", /*type=*/_28, /*offset=*/128);
    const auto* _29 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"tag", /*type=*/_29, /*offset=*/160);
    const auto* _31 =
        BuildPJType<::v0_1::pj::reflect::TermAttribute>::build(ctx, domain);
    const auto* _30 = PJCreateVectorType(
        ctx, /*elem=*/_31, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] = PJCreateStructField(/*name=*/"attributes", /*type=*/_30,
                                    /*offset=*/224);
    const char* _32[3] = {"pj", "reflect", "Term"};
    const PJStructType* _33 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_32, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/352, /*alignment=*/8);
    return _33;
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
    const auto* _36 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _35 = PJCreateVectorType(
        ctx, /*elem=*/_36, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _34 = PJCreateVectorType(
        ctx, /*elem=*/_35, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_34, /*offset=*/0);
    const auto* _38 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _37 = PJCreateVectorType(
        ctx, /*elem=*/_38, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_37, /*offset=*/128);
    const auto* _39 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_39,
                                    /*offset=*/256);
    const auto* _40 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_40,
                                    /*offset=*/320);
    const auto* _41 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] =
        PJCreateStructField(/*name=*/"term_size", /*type=*/_41, /*offset=*/384);
    const auto* _42 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"tag_offset", /*type=*/_42,
                                    /*offset=*/448);
    const auto* _43 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_43, /*offset=*/512);
    const auto* _44 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"size", /*type=*/_44, /*offset=*/576);
    const auto* _45 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_45, /*offset=*/640);
    const char* _46[3] = {"pj", "reflect", "InlineVariant"};
    const PJStructType* _47 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_46, /*type_domain=*/domain,
        /*num_fields=*/9, /*fields=*/fields, /*size=*/704, /*alignment=*/8);
    return _47;
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
    const PJUnitType* _48 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kSigned", /*type=*/_48, /*tag=*/1);
    const PJUnitType* _49 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kSignless", /*type=*/_49, /*tag=*/3);
    const PJUnitType* _50 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"kUnsigned", /*type=*/_50, /*tag=*/2);
    const PJUnitType* _51 = PJCreateUnitType(ctx);
    terms[3] = PJCreateTerm(/*name=*/"undef", /*type=*/_51, /*tag=*/0);
    const char* _52[2] = {"pj", "Sign"};
    const PJInlineVariantType* _53 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_52, /*type_domain=*/domain,
        /*num_terms=*/4, /*terms=*/terms, /*default_term=*/3, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _53;
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
    const auto* _54 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_54, /*offset=*/0);
    const auto* _55 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_55, /*offset=*/64);
    const auto* _56 = BuildPJType<::v0_1::pj::Sign>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"sign", /*type=*/_56, /*offset=*/128);
    const char* _57[3] = {"pj", "reflect", "Int"};
    const PJStructType* _58 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_57, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/136, /*alignment=*/8);
    return _58;
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
    const auto* _61 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _60 = PJCreateVectorType(
        ctx, /*elem=*/_61, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _59 = PJCreateVectorType(
        ctx, /*elem=*/_60, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_59, /*offset=*/0);
    const auto* _63 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _62 = PJCreateVectorType(
        ctx, /*elem=*/_63, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_62, /*offset=*/128);
    const auto* _64 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_64,
                                    /*offset=*/256);
    const auto* _65 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_65, /*offset=*/320);
    const auto* _66 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"tag_alignment", /*type=*/_66,
                                    /*offset=*/384);
    const auto* _67 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_67,
                                    /*offset=*/448);
    const auto* _68 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"term_alignment", /*type=*/_68,
                                    /*offset=*/512);
    const char* _69[3] = {"pj", "reflect", "OutlineVariant"};
    const PJStructType* _70 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_69, /*type_domain=*/domain,
        /*num_fields=*/7, /*fields=*/fields, /*size=*/576, /*alignment=*/8);
    return _70;
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
    const auto* _71 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_71, /*offset=*/0);
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
    fields[1] =
        PJCreateStructField(/*name=*/"name", /*type=*/_72, /*offset=*/32);
    const auto* _74 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"offset", /*type=*/_74, /*offset=*/160);
    const char* _75[3] = {"pj", "reflect", "StructField"};
    const PJStructType* _76 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_75, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _76;
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
    const auto* _79 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _78 = PJCreateVectorType(
        ctx, /*elem=*/_79, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _77 = PJCreateVectorType(
        ctx, /*elem=*/_78, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_77, /*offset=*/0);
    const auto* _81 =
        BuildPJType<::v0_1::pj::reflect::StructField>::build(ctx, domain);
    const auto* _80 = PJCreateVectorType(
        ctx, /*elem=*/_81, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"fields", /*type=*/_80, /*offset=*/128);
    const auto* _82 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"size", /*type=*/_82, /*offset=*/256);
    const auto* _83 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_83, /*offset=*/320);
    const char* _84[3] = {"pj", "reflect", "Struct"};
    const PJStructType* _85 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_84, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/384, /*alignment=*/8);
    return _85;
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
    const char* _86[3] = {"pj", "reflect", "Unit"};
    const PJStructType* _87 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_86, /*type_domain=*/domain,
        /*num_fields=*/0, /*fields=*/fields, /*size=*/0, /*alignment=*/8);
    return _87;
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
    const PJUnitType* _88 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOffset", /*type=*/_88, /*tag=*/2);
    const PJUnitType* _89 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kPointer", /*type=*/_89, /*tag=*/1);
    const PJUnitType* _90 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_90, /*tag=*/0);
    const char* _91[2] = {"pj", "ReferenceMode"};
    const PJInlineVariantType* _92 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_91, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _92;
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
    const auto* _93 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"elem", /*type=*/_93, /*offset=*/0);
    const auto* _94 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"elem_width", /*type=*/_94, /*offset=*/32);
    const auto* _95 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"min_length", /*type=*/_95, /*offset=*/96);
    const auto* _96 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[3] = PJCreateStructField(/*name=*/"max_length", /*type=*/_96,
                                    /*offset=*/160);
    const auto* _97 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[4] =
        PJCreateStructField(/*name=*/"ppl_count", /*type=*/_97, /*offset=*/224);
    const auto* _98 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"length_offset", /*type=*/_98,
                                    /*offset=*/288);
    const auto* _99 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"length_size", /*type=*/_99,
                                    /*offset=*/352);
    const auto* _100 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] = PJCreateStructField(/*name=*/"ref_offset", /*type=*/_100,
                                    /*offset=*/416);
    const auto* _101 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"ref_size", /*type=*/_101, /*offset=*/480);
    const auto* _102 =
        BuildPJType<::v0_1::pj::ReferenceMode>::build(ctx, domain);
    fields[9] = PJCreateStructField(/*name=*/"reference_mode", /*type=*/_102,
                                    /*offset=*/544);
    const auto* _103 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[10] = PJCreateStructField(/*name=*/"inline_payload_offset",
                                     /*type=*/_103, /*offset=*/552);
    const auto* _104 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[11] = PJCreateStructField(/*name=*/"partial_payload_offset",
                                     /*type=*/_104, /*offset=*/616);
    const auto* _105 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[12] =
        PJCreateStructField(/*name=*/"size", /*type=*/_105, /*offset=*/680);
    const auto* _106 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[13] = PJCreateStructField(/*name=*/"alignment", /*type=*/_106,
                                     /*offset=*/744);
    const char* _107[3] = {"pj", "reflect", "Vector"};
    const PJStructType* _108 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_107, /*type_domain=*/domain,
        /*num_fields=*/14, /*fields=*/fields, /*size=*/808, /*alignment=*/8);
    return _108;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Type> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[8];
    const auto* _109 =
        BuildPJType<::v0_1::pj::reflect::Array>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"Array", /*type=*/_109, /*tag=*/6);
    const auto* _110 =
        BuildPJType<::v0_1::pj::reflect::InlineVariant>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"InlineVariant", /*type=*/_110, /*tag=*/4);
    const auto* _111 =
        BuildPJType<::v0_1::pj::reflect::Int>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"Int", /*type=*/_111, /*tag=*/1);
    const auto* _112 =
        BuildPJType<::v0_1::pj::reflect::OutlineVariant>::build(ctx, domain);
    terms[3] =
        PJCreateTerm(/*name=*/"OutlineVariant", /*type=*/_112, /*tag=*/5);
    const auto* _113 =
        BuildPJType<::v0_1::pj::reflect::Struct>::build(ctx, domain);
    terms[4] = PJCreateTerm(/*name=*/"Struct", /*type=*/_113, /*tag=*/3);
    const auto* _114 =
        BuildPJType<::v0_1::pj::reflect::Unit>::build(ctx, domain);
    terms[5] = PJCreateTerm(/*name=*/"Unit", /*type=*/_114, /*tag=*/2);
    const auto* _115 =
        BuildPJType<::v0_1::pj::reflect::Vector>::build(ctx, domain);
    terms[6] = PJCreateTerm(/*name=*/"Vector", /*type=*/_115, /*tag=*/7);
    const PJUnitType* _116 = PJCreateUnitType(ctx);
    terms[7] = PJCreateTerm(/*name=*/"undef", /*type=*/_116, /*tag=*/0);
    const char* _117[3] = {"pj", "reflect", "Type"};
    const PJInlineVariantType* _118 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_117, /*type_domain=*/domain,
        /*num_terms=*/8, /*terms=*/terms, /*default_term=*/7, /*term_offset=*/0,
        /*term_size=*/808, /*tag_offset=*/808, /*tag_width=*/8, /*size=*/816,
        /*alignment=*/8);
    return _118;
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
    const auto* _119 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"pj_version", /*type=*/_119, /*offset=*/0);
    const auto* _120 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"head", /*type=*/_120, /*offset=*/32);
    const auto* _121 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"buffer_offset", /*type=*/_121,
                                    /*offset=*/64);
    const auto* _123 =
        BuildPJType<::v0_1::pj::reflect::Type>::build(ctx, domain);
    const auto* _122 = PJCreateVectorType(
        ctx, /*elem=*/_123, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] =
        PJCreateStructField(/*name=*/"types", /*type=*/_122, /*offset=*/128);
    const char* _124[3] = {"pj", "reflect", "Protocol"};
    const PJStructType* _125 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_124, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/256, /*alignment=*/8);
    return _125;
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
    const auto* _126 =
        BuildPJType<::v0_1::pj::reflect::Protocol>::build(ctx, domain);
    return PJCreateProtocolType(ctx, _126, 0);
  }
};
}  // namespace gen

}  // namespace pj
