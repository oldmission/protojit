#pragma once
#include <cstddef>
#include <string_view>

#include "pj/runtime.hpp"
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
struct BuildPJType<::v0_2::pj::reflect::ShortIntType> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _9 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOriginal", /*type=*/_9, /*tag=*/2);
    const PJUnitType* _10 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kShort", /*type=*/_10, /*tag=*/1);
    const PJUnitType* _11 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_11, /*tag=*/0);
    const char* _12[3] = {"pj", "reflect", "ShortIntType"};
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
struct BuildPJType<::v0_2::pj::reflect::ShortInt> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[4];
    const auto* _14 =
        BuildPJType<::v0_2::pj::reflect::ShortIntType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_14, /*offset=*/0);
    const auto* _15 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"threshold", /*type=*/_15, /*offset=*/8);
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
    const char* _20[3] = {"pj", "reflect", "ShortInt"};
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
struct BuildPJType<::v0_2::pj::reflect::Undef> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[1];
    const auto* _22 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"is_default", /*type=*/_22, /*offset=*/0);
    const char* _23[3] = {"pj", "reflect", "Undef"};
    const PJStructType* _24 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_23, /*type_domain=*/domain,
        /*num_fields=*/1, /*fields=*/fields, /*size=*/8, /*alignment=*/8);
    return _24;
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
    const PJUnitType* _25 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kInline", /*type=*/_25, /*tag=*/1);
    const PJUnitType* _26 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kOutline", /*type=*/_26, /*tag=*/2);
    const PJUnitType* _27 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_27, /*tag=*/0);
    const char* _28[3] = {"pj", "reflect", "VectorSplitType"};
    const PJInlineVariantType* _29 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_28, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _29;
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
    const auto* _30 =
        BuildPJType<::v0_2::pj::reflect::VectorSplitType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_30, /*offset=*/0);
    const auto* _31 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] = PJCreateStructField(/*name=*/"inline_length", /*type=*/_31,
                                    /*offset=*/8);
    const auto* _34 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _33 = PJCreateVectorType(
        ctx, /*elem=*/_34, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _32 = PJCreateVectorType(
        ctx, /*elem=*/_33, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_32, /*offset=*/72);
    const auto* _35 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_35,
                                    /*offset=*/200);
    const char* _36[3] = {"pj", "reflect", "VectorSplit"};
    const PJStructType* _37 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_36, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _37;
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
    const auto* _38 =
        BuildPJType<::v0_2::pj::reflect::ShortInt>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"short_int", /*type=*/_38, /*tag=*/2);
    const auto* _39 =
        BuildPJType<::v0_2::pj::reflect::Undef>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"undef", /*type=*/_39, /*tag=*/0);
    const auto* _40 =
        BuildPJType<::v0_2::pj::reflect::VectorSplit>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"vector_split", /*type=*/_40, /*tag=*/1);
    const char* _41[3] = {"pj", "reflect", "TermAttribute"};
    const PJInlineVariantType* _42 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_41, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/1, /*term_offset=*/0,
        /*term_size=*/208, /*tag_offset=*/208, /*tag_width=*/8, /*size=*/216,
        /*alignment=*/8);
    return _42;
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
    const auto* _44 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _43 = PJCreateVectorType(
        ctx, /*elem=*/_44, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_43, /*offset=*/0);
    const auto* _45 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"type", /*type=*/_45, /*offset=*/128);
    const auto* _46 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"tag", /*type=*/_46, /*offset=*/160);
    const auto* _48 =
        BuildPJType<::v0_2::pj::reflect::TermAttribute>::build(ctx, domain);
    const auto* _47 = PJCreateVectorType(
        ctx, /*elem=*/_48, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] = PJCreateStructField(/*name=*/"attributes", /*type=*/_47,
                                    /*offset=*/224);
    const char* _49[3] = {"pj", "reflect", "Term"};
    const PJStructType* _50 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_49, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/352, /*alignment=*/8);
    return _50;
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
    const auto* _51 = PJCreateVectorType(
        ctx, /*elem=*/_52, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_51, /*offset=*/0);
    const auto* _55 =
        BuildPJType<::v0_2::pj::reflect::Term>::build(ctx, domain);
    const auto* _54 = PJCreateVectorType(
        ctx, /*elem=*/_55, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_54, /*offset=*/128);
    const auto* _56 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_56,
                                    /*offset=*/256);
    const auto* _57 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_57,
                                    /*offset=*/320);
    const auto* _58 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[4] =
        PJCreateStructField(/*name=*/"term_size", /*type=*/_58, /*offset=*/384);
    const auto* _59 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"tag_offset", /*type=*/_59,
                                    /*offset=*/448);
    const auto* _60 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[6] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_60, /*offset=*/512);
    const auto* _61 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"size", /*type=*/_61, /*offset=*/576);
    const auto* _62 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_62, /*offset=*/640);
    const char* _63[3] = {"pj", "reflect", "InlineVariant"};
    const PJStructType* _64 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_63, /*type_domain=*/domain,
        /*num_fields=*/9, /*fields=*/fields, /*size=*/704, /*alignment=*/8);
    return _64;
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
    const PJUnitType* _65 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kSigned", /*type=*/_65, /*tag=*/1);
    const PJUnitType* _66 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kSignless", /*type=*/_66, /*tag=*/3);
    const PJUnitType* _67 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"kUnsigned", /*type=*/_67, /*tag=*/2);
    const PJUnitType* _68 = PJCreateUnitType(ctx);
    terms[3] = PJCreateTerm(/*name=*/"undef", /*type=*/_68, /*tag=*/0);
    const char* _69[2] = {"pj", "Sign"};
    const PJInlineVariantType* _70 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_69, /*type_domain=*/domain,
        /*num_terms=*/4, /*terms=*/terms, /*default_term=*/3, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _70;
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
    const auto* _71 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_71, /*offset=*/0);
    const auto* _72 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_72, /*offset=*/64);
    const auto* _73 = BuildPJType<::v0_2::pj::Sign>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"sign", /*type=*/_73, /*offset=*/128);
    const char* _74[3] = {"pj", "reflect", "Int"};
    const PJStructType* _75 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_74, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/136, /*alignment=*/8);
    return _75;
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
    const auto* _78 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _77 = PJCreateVectorType(
        ctx, /*elem=*/_78, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _76 = PJCreateVectorType(
        ctx, /*elem=*/_77, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_76, /*offset=*/0);
    const auto* _80 =
        BuildPJType<::v0_2::pj::reflect::Term>::build(ctx, domain);
    const auto* _79 = PJCreateVectorType(
        ctx, /*elem=*/_80, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_79, /*offset=*/128);
    const auto* _81 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_81,
                                    /*offset=*/256);
    const auto* _82 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_82, /*offset=*/320);
    const auto* _83 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"tag_alignment", /*type=*/_83,
                                    /*offset=*/384);
    const auto* _84 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_84,
                                    /*offset=*/448);
    const auto* _85 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"term_alignment", /*type=*/_85,
                                    /*offset=*/512);
    const char* _86[3] = {"pj", "reflect", "OutlineVariant"};
    const PJStructType* _87 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_86, /*type_domain=*/domain,
        /*num_fields=*/7, /*fields=*/fields, /*size=*/576, /*alignment=*/8);
    return _87;
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
    const auto* _88 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_88, /*offset=*/0);
    const auto* _90 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _89 = PJCreateVectorType(
        ctx, /*elem=*/_90, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"name", /*type=*/_89, /*offset=*/32);
    const auto* _91 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"offset", /*type=*/_91, /*offset=*/160);
    const char* _92[3] = {"pj", "reflect", "StructField"};
    const PJStructType* _93 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_92, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _93;
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
    const auto* _96 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _95 = PJCreateVectorType(
        ctx, /*elem=*/_96, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _94 = PJCreateVectorType(
        ctx, /*elem=*/_95, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_94, /*offset=*/0);
    const auto* _98 =
        BuildPJType<::v0_2::pj::reflect::StructField>::build(ctx, domain);
    const auto* _97 = PJCreateVectorType(
        ctx, /*elem=*/_98, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"fields", /*type=*/_97, /*offset=*/128);
    const auto* _99 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"size", /*type=*/_99, /*offset=*/256);
    const auto* _100 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"alignment", /*type=*/_100,
                                    /*offset=*/320);
    const char* _101[3] = {"pj", "reflect", "Struct"};
    const PJStructType* _102 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_101, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/384, /*alignment=*/8);
    return _102;
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
    const char* _103[3] = {"pj", "reflect", "Unit"};
    const PJStructType* _104 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_103, /*type_domain=*/domain,
        /*num_fields=*/0, /*fields=*/fields, /*size=*/0, /*alignment=*/8);
    return _104;
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
    const PJUnitType* _105 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOffset", /*type=*/_105, /*tag=*/2);
    const PJUnitType* _106 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kPointer", /*type=*/_106, /*tag=*/1);
    const PJUnitType* _107 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_107, /*tag=*/0);
    const char* _108[2] = {"pj", "ReferenceMode"};
    const PJInlineVariantType* _109 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_108, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _109;
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
    const auto* _110 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"elem", /*type=*/_110, /*offset=*/0);
    const auto* _111 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[1] = PJCreateStructField(/*name=*/"elem_width", /*type=*/_111,
                                    /*offset=*/32);
    const auto* _112 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"min_length", /*type=*/_112,
                                    /*offset=*/96);
    const auto* _113 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[3] = PJCreateStructField(/*name=*/"max_length", /*type=*/_113,
                                    /*offset=*/160);
    const auto* _114 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[4] = PJCreateStructField(/*name=*/"ppl_count", /*type=*/_114,
                                    /*offset=*/224);
    const auto* _115 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"length_offset", /*type=*/_115,
                                    /*offset=*/288);
    const auto* _116 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"length_size", /*type=*/_116,
                                    /*offset=*/352);
    const auto* _117 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[7] = PJCreateStructField(/*name=*/"ref_offset", /*type=*/_117,
                                    /*offset=*/416);
    const auto* _118 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"ref_size", /*type=*/_118, /*offset=*/480);
    const auto* _119 =
        BuildPJType<::v0_2::pj::ReferenceMode>::build(ctx, domain);
    fields[9] = PJCreateStructField(/*name=*/"reference_mode", /*type=*/_119,
                                    /*offset=*/544);
    const auto* _120 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[10] = PJCreateStructField(/*name=*/"inline_payload_offset",
                                     /*type=*/_120, /*offset=*/552);
    const auto* _121 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[11] = PJCreateStructField(/*name=*/"partial_payload_offset",
                                     /*type=*/_121, /*offset=*/616);
    const auto* _122 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[12] =
        PJCreateStructField(/*name=*/"size", /*type=*/_122, /*offset=*/680);
    const auto* _123 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[13] = PJCreateStructField(/*name=*/"alignment", /*type=*/_123,
                                     /*offset=*/744);
    const char* _124[3] = {"pj", "reflect", "Vector"};
    const PJStructType* _125 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_124, /*type_domain=*/domain,
        /*num_fields=*/14, /*fields=*/fields, /*size=*/808, /*alignment=*/8);
    return _125;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_2::pj::reflect::Type> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[8];
    const auto* _126 =
        BuildPJType<::v0_2::pj::reflect::Array>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"Array", /*type=*/_126, /*tag=*/6);
    const auto* _127 =
        BuildPJType<::v0_2::pj::reflect::InlineVariant>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"InlineVariant", /*type=*/_127, /*tag=*/4);
    const auto* _128 =
        BuildPJType<::v0_2::pj::reflect::Int>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"Int", /*type=*/_128, /*tag=*/1);
    const auto* _129 =
        BuildPJType<::v0_2::pj::reflect::OutlineVariant>::build(ctx, domain);
    terms[3] =
        PJCreateTerm(/*name=*/"OutlineVariant", /*type=*/_129, /*tag=*/5);
    const auto* _130 =
        BuildPJType<::v0_2::pj::reflect::Struct>::build(ctx, domain);
    terms[4] = PJCreateTerm(/*name=*/"Struct", /*type=*/_130, /*tag=*/3);
    const auto* _131 =
        BuildPJType<::v0_2::pj::reflect::Unit>::build(ctx, domain);
    terms[5] = PJCreateTerm(/*name=*/"Unit", /*type=*/_131, /*tag=*/2);
    const auto* _132 =
        BuildPJType<::v0_2::pj::reflect::Vector>::build(ctx, domain);
    terms[6] = PJCreateTerm(/*name=*/"Vector", /*type=*/_132, /*tag=*/7);
    const PJUnitType* _133 = PJCreateUnitType(ctx);
    terms[7] = PJCreateTerm(/*name=*/"undef", /*type=*/_133, /*tag=*/0);
    const char* _134[3] = {"pj", "reflect", "Type"};
    const PJInlineVariantType* _135 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_134, /*type_domain=*/domain,
        /*num_terms=*/8, /*terms=*/terms, /*default_term=*/7, /*term_offset=*/0,
        /*term_size=*/808, /*tag_offset=*/808, /*tag_width=*/8, /*size=*/816,
        /*alignment=*/8);
    return _135;
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
    const auto* _136 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"pj_version", /*type=*/_136, /*offset=*/0);
    const auto* _137 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"head", /*type=*/_137, /*offset=*/32);
    const auto* _138 = BuildPJType<::v0_2::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"buffer_offset", /*type=*/_138,
                                    /*offset=*/64);
    const auto* _140 =
        BuildPJType<::v0_2::pj::reflect::Type>::build(ctx, domain);
    const auto* _139 = PJCreateVectorType(
        ctx, /*elem=*/_140, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] =
        PJCreateStructField(/*name=*/"types", /*type=*/_139, /*offset=*/128);
    const char* _141[3] = {"pj", "reflect", "Protocol"};
    const PJStructType* _142 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_141, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/256, /*alignment=*/8);
    return _142;
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
    const auto* _143 =
        BuildPJType<::v0_2::pj::reflect::Protocol>::build(ctx, domain);
    return PJCreateProtocolType(ctx, _143, 0);
  }
};
}  // namespace gen

}  // namespace pj
