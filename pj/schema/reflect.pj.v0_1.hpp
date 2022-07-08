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
struct BuildPJType<::v0_1::pj::reflect::Undef> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[1];
    const auto* _9 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                     /*sign=*/PJ_SIGN_UNSIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"is_default", /*type=*/_9, /*offset=*/0);
    const char* _10[3] = {"pj", "reflect", "Undef"};
    const PJStructType* _11 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_10, /*type_domain=*/domain,
        /*num_fields=*/1, /*fields=*/fields, /*size=*/8, /*alignment=*/8);
    return _11;
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
    const PJUnitType* _12 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kInline", /*type=*/_12, /*tag=*/1);
    const PJUnitType* _13 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kOutline", /*type=*/_13, /*tag=*/2);
    const PJUnitType* _14 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_14, /*tag=*/0);
    const char* _15[3] = {"pj", "reflect", "VectorSplitType"};
    const PJInlineVariantType* _16 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_15, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _16;
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
    const auto* _17 =
        BuildPJType<::v0_1::pj::reflect::VectorSplitType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_17, /*offset=*/0);
    const auto* _18 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] = PJCreateStructField(/*name=*/"inline_length", /*type=*/_18,
                                    /*offset=*/8);
    const auto* _21 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _20 = PJCreateVectorType(
        ctx, /*elem=*/_21, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _19 = PJCreateVectorType(
        ctx, /*elem=*/_20, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_19, /*offset=*/72);
    const auto* _22 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_22,
                                    /*offset=*/200);
    const char* _23[3] = {"pj", "reflect", "VectorSplit"};
    const PJStructType* _24 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_23, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _24;
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
    const auto* _25 =
        BuildPJType<::v0_1::pj::reflect::Undef>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"undef", /*type=*/_25, /*tag=*/0);
    const auto* _26 =
        BuildPJType<::v0_1::pj::reflect::VectorSplit>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"vector_split", /*type=*/_26, /*tag=*/1);
    const char* _27[3] = {"pj", "reflect", "TermAttribute"};
    const PJInlineVariantType* _28 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_27, /*type_domain=*/domain,
        /*num_terms=*/2, /*terms=*/terms, /*default_term=*/0, /*term_offset=*/0,
        /*term_size=*/208, /*tag_offset=*/208, /*tag_width=*/8, /*size=*/216,
        /*alignment=*/8);
    return _28;
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
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_29, /*offset=*/0);
    const auto* _31 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"type", /*type=*/_31, /*offset=*/128);
    const auto* _32 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"tag", /*type=*/_32, /*offset=*/160);
    const auto* _34 =
        BuildPJType<::v0_1::pj::reflect::TermAttribute>::build(ctx, domain);
    const auto* _33 = PJCreateVectorType(
        ctx, /*elem=*/_34, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] = PJCreateStructField(/*name=*/"attributes", /*type=*/_33,
                                    /*offset=*/224);
    const char* _35[3] = {"pj", "reflect", "Term"};
    const PJStructType* _36 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_35, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/352, /*alignment=*/8);
    return _36;
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
    const auto* _37 = PJCreateVectorType(
        ctx, /*elem=*/_38, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_37, /*offset=*/0);
    const auto* _41 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _40 = PJCreateVectorType(
        ctx, /*elem=*/_41, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_40, /*offset=*/128);
    const auto* _42 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_42,
                                    /*offset=*/256);
    const auto* _43 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_43,
                                    /*offset=*/320);
    const auto* _44 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] =
        PJCreateStructField(/*name=*/"term_size", /*type=*/_44, /*offset=*/384);
    const auto* _45 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"tag_offset", /*type=*/_45,
                                    /*offset=*/448);
    const auto* _46 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_46, /*offset=*/512);
    const auto* _47 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"size", /*type=*/_47, /*offset=*/576);
    const auto* _48 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_48, /*offset=*/640);
    const char* _49[3] = {"pj", "reflect", "InlineVariant"};
    const PJStructType* _50 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_49, /*type_domain=*/domain,
        /*num_fields=*/9, /*fields=*/fields, /*size=*/704, /*alignment=*/8);
    return _50;
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
    const PJUnitType* _51 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kSigned", /*type=*/_51, /*tag=*/1);
    const PJUnitType* _52 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kSignless", /*type=*/_52, /*tag=*/3);
    const PJUnitType* _53 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"kUnsigned", /*type=*/_53, /*tag=*/2);
    const PJUnitType* _54 = PJCreateUnitType(ctx);
    terms[3] = PJCreateTerm(/*name=*/"undef", /*type=*/_54, /*tag=*/0);
    const char* _55[2] = {"pj", "Sign"};
    const PJInlineVariantType* _56 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_55, /*type_domain=*/domain,
        /*num_terms=*/4, /*terms=*/terms, /*default_term=*/3, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _56;
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
    const auto* _57 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_57, /*offset=*/0);
    const auto* _58 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_58, /*offset=*/64);
    const auto* _59 = BuildPJType<::v0_1::pj::Sign>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"sign", /*type=*/_59, /*offset=*/128);
    const char* _60[3] = {"pj", "reflect", "Int"};
    const PJStructType* _61 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_60, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/136, /*alignment=*/8);
    return _61;
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
    const auto* _64 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _63 = PJCreateVectorType(
        ctx, /*elem=*/_64, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _62 = PJCreateVectorType(
        ctx, /*elem=*/_63, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_62, /*offset=*/0);
    const auto* _66 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _65 = PJCreateVectorType(
        ctx, /*elem=*/_66, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_65, /*offset=*/128);
    const auto* _67 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] = PJCreateStructField(/*name=*/"default_term", /*type=*/_67,
                                    /*offset=*/256);
    const auto* _68 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_68, /*offset=*/320);
    const auto* _69 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"tag_alignment", /*type=*/_69,
                                    /*offset=*/384);
    const auto* _70 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_70,
                                    /*offset=*/448);
    const auto* _71 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"term_alignment", /*type=*/_71,
                                    /*offset=*/512);
    const char* _72[3] = {"pj", "reflect", "OutlineVariant"};
    const PJStructType* _73 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_72, /*type_domain=*/domain,
        /*num_fields=*/7, /*fields=*/fields, /*size=*/576, /*alignment=*/8);
    return _73;
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
    const auto* _74 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_74, /*offset=*/0);
    const auto* _76 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _75 = PJCreateVectorType(
        ctx, /*elem=*/_76, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"name", /*type=*/_75, /*offset=*/32);
    const auto* _77 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"offset", /*type=*/_77, /*offset=*/160);
    const char* _78[3] = {"pj", "reflect", "StructField"};
    const PJStructType* _79 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_78, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _79;
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
    const auto* _82 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _81 = PJCreateVectorType(
        ctx, /*elem=*/_82, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _80 = PJCreateVectorType(
        ctx, /*elem=*/_81, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_80, /*offset=*/0);
    const auto* _84 =
        BuildPJType<::v0_1::pj::reflect::StructField>::build(ctx, domain);
    const auto* _83 = PJCreateVectorType(
        ctx, /*elem=*/_84, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"fields", /*type=*/_83, /*offset=*/128);
    const auto* _85 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"size", /*type=*/_85, /*offset=*/256);
    const auto* _86 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_86, /*offset=*/320);
    const char* _87[3] = {"pj", "reflect", "Struct"};
    const PJStructType* _88 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_87, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/384, /*alignment=*/8);
    return _88;
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
    const char* _89[3] = {"pj", "reflect", "Unit"};
    const PJStructType* _90 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_89, /*type_domain=*/domain,
        /*num_fields=*/0, /*fields=*/fields, /*size=*/0, /*alignment=*/8);
    return _90;
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
    const PJUnitType* _91 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOffset", /*type=*/_91, /*tag=*/2);
    const PJUnitType* _92 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kPointer", /*type=*/_92, /*tag=*/1);
    const PJUnitType* _93 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"undef", /*type=*/_93, /*tag=*/0);
    const char* _94[2] = {"pj", "ReferenceMode"};
    const PJInlineVariantType* _95 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_94, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*default_term=*/2, /*term_offset=*/0,
        /*term_size=*/0, /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8,
        /*alignment=*/8);
    return _95;
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
    const auto* _96 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"elem", /*type=*/_96, /*offset=*/0);
    const auto* _97 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"elem_width", /*type=*/_97, /*offset=*/32);
    const auto* _98 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"min_length", /*type=*/_98, /*offset=*/96);
    const auto* _99 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[3] = PJCreateStructField(/*name=*/"max_length", /*type=*/_99,
                                    /*offset=*/160);
    const auto* _100 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[4] = PJCreateStructField(/*name=*/"ppl_count", /*type=*/_100,
                                    /*offset=*/224);
    const auto* _101 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"length_offset", /*type=*/_101,
                                    /*offset=*/288);
    const auto* _102 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"length_size", /*type=*/_102,
                                    /*offset=*/352);
    const auto* _103 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] = PJCreateStructField(/*name=*/"ref_offset", /*type=*/_103,
                                    /*offset=*/416);
    const auto* _104 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[8] =
        PJCreateStructField(/*name=*/"ref_size", /*type=*/_104, /*offset=*/480);
    const auto* _105 =
        BuildPJType<::v0_1::pj::ReferenceMode>::build(ctx, domain);
    fields[9] = PJCreateStructField(/*name=*/"reference_mode", /*type=*/_105,
                                    /*offset=*/544);
    const auto* _106 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[10] = PJCreateStructField(/*name=*/"inline_payload_offset",
                                     /*type=*/_106, /*offset=*/552);
    const auto* _107 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[11] = PJCreateStructField(/*name=*/"partial_payload_offset",
                                     /*type=*/_107, /*offset=*/616);
    const auto* _108 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[12] =
        PJCreateStructField(/*name=*/"size", /*type=*/_108, /*offset=*/680);
    const auto* _109 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[13] = PJCreateStructField(/*name=*/"alignment", /*type=*/_109,
                                     /*offset=*/744);
    const char* _110[3] = {"pj", "reflect", "Vector"};
    const PJStructType* _111 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_110, /*type_domain=*/domain,
        /*num_fields=*/14, /*fields=*/fields, /*size=*/808, /*alignment=*/8);
    return _111;
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
    const auto* _112 =
        BuildPJType<::v0_1::pj::reflect::Array>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"Array", /*type=*/_112, /*tag=*/6);
    const auto* _113 =
        BuildPJType<::v0_1::pj::reflect::InlineVariant>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"InlineVariant", /*type=*/_113, /*tag=*/4);
    const auto* _114 =
        BuildPJType<::v0_1::pj::reflect::Int>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"Int", /*type=*/_114, /*tag=*/1);
    const auto* _115 =
        BuildPJType<::v0_1::pj::reflect::OutlineVariant>::build(ctx, domain);
    terms[3] =
        PJCreateTerm(/*name=*/"OutlineVariant", /*type=*/_115, /*tag=*/5);
    const auto* _116 =
        BuildPJType<::v0_1::pj::reflect::Struct>::build(ctx, domain);
    terms[4] = PJCreateTerm(/*name=*/"Struct", /*type=*/_116, /*tag=*/3);
    const auto* _117 =
        BuildPJType<::v0_1::pj::reflect::Unit>::build(ctx, domain);
    terms[5] = PJCreateTerm(/*name=*/"Unit", /*type=*/_117, /*tag=*/2);
    const auto* _118 =
        BuildPJType<::v0_1::pj::reflect::Vector>::build(ctx, domain);
    terms[6] = PJCreateTerm(/*name=*/"Vector", /*type=*/_118, /*tag=*/7);
    const PJUnitType* _119 = PJCreateUnitType(ctx);
    terms[7] = PJCreateTerm(/*name=*/"undef", /*type=*/_119, /*tag=*/0);
    const char* _120[3] = {"pj", "reflect", "Type"};
    const PJInlineVariantType* _121 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_120, /*type_domain=*/domain,
        /*num_terms=*/8, /*terms=*/terms, /*default_term=*/7, /*term_offset=*/0,
        /*term_size=*/808, /*tag_offset=*/808, /*tag_width=*/8, /*size=*/816,
        /*alignment=*/8);
    return _121;
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
    const auto* _122 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"pj_version", /*type=*/_122, /*offset=*/0);
    const auto* _123 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"head", /*type=*/_123, /*offset=*/32);
    const auto* _124 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"buffer_offset", /*type=*/_124,
                                    /*offset=*/64);
    const auto* _126 =
        BuildPJType<::v0_1::pj::reflect::Type>::build(ctx, domain);
    const auto* _125 = PJCreateVectorType(
        ctx, /*elem=*/_126, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] =
        PJCreateStructField(/*name=*/"types", /*type=*/_125, /*offset=*/128);
    const char* _127[3] = {"pj", "reflect", "Protocol"};
    const PJStructType* _128 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_127, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/256, /*alignment=*/8);
    return _128;
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
    const auto* _129 =
        BuildPJType<::v0_1::pj::reflect::Protocol>::build(ctx, domain);
    return PJCreateProtocolType(ctx, _129, 0);
  }
};
}  // namespace gen

}  // namespace pj
