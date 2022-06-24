#pragma once
#include <cstddef>
#include <string_view>
#include "pj/protojit.hpp"
#include "pj/runtime.hpp"

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
    const PJTerm* terms[2];
    const PJUnitType* _9 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kInline", /*type=*/_9, /*tag=*/1);
    const PJUnitType* _10 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kOutline", /*type=*/_10, /*tag=*/2);
    const char* _11[3] = {"pj", "reflect", "VectorSplitType"};
    const PJInlineVariantType* _12 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_11, /*type_domain=*/domain,
        /*num_terms=*/2, /*terms=*/terms, /*term_offset=*/0, /*term_size=*/0,
        /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8, /*alignment=*/8);
    return _12;
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
    const auto* _13 =
        BuildPJType<::v0_1::pj::reflect::VectorSplitType>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_13, /*offset=*/0);
    const auto* _14 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] = PJCreateStructField(/*name=*/"inline_length", /*type=*/_14,
                                    /*offset=*/8);
    const auto* _17 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _16 = PJCreateVectorType(
        ctx, /*elem=*/_17, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _15 = PJCreateVectorType(
        ctx, /*elem=*/_16, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[2] =
        PJCreateStructField(/*name=*/"path", /*type=*/_15, /*offset=*/72);
    const auto* _18 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[3] = PJCreateStructField(/*name=*/"is_default", /*type=*/_18,
                                    /*offset=*/200);
    const char* _19[3] = {"pj", "reflect", "VectorSplit"};
    const PJStructType* _20 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_19, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/208, /*alignment=*/8);
    return _20;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::TermAttribute> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[1];
    const auto* _21 =
        BuildPJType<::v0_1::pj::reflect::VectorSplit>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"vector_split", /*type=*/_21, /*tag=*/1);
    const char* _22[3] = {"pj", "reflect", "TermAttribute"};
    const PJInlineVariantType* _23 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_22, /*type_domain=*/domain,
        /*num_terms=*/1, /*terms=*/terms, /*term_offset=*/0, /*term_size=*/208,
        /*tag_offset=*/208, /*tag_width=*/8, /*size=*/216, /*alignment=*/8);
    return _23;
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
    const auto* _25 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _24 = PJCreateVectorType(
        ctx, /*elem=*/_25, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_24, /*offset=*/0);
    const auto* _26 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"type", /*type=*/_26, /*offset=*/128);
    const auto* _27 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"tag", /*type=*/_27, /*offset=*/160);
    const auto* _29 =
        BuildPJType<::v0_1::pj::reflect::TermAttribute>::build(ctx, domain);
    const auto* _28 = PJCreateVectorType(
        ctx, /*elem=*/_29, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] = PJCreateStructField(/*name=*/"attributes", /*type=*/_28,
                                    /*offset=*/224);
    const char* _30[3] = {"pj", "reflect", "Term"};
    const PJStructType* _31 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_30, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/352, /*alignment=*/8);
    return _31;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::InlineVariant> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[8];
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
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_32, /*offset=*/0);
    const auto* _36 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _35 = PJCreateVectorType(
        ctx, /*elem=*/_36, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_35, /*offset=*/128);
    const auto* _37 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_37,
                                    /*offset=*/256);
    const auto* _38 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"term_size", /*type=*/_38, /*offset=*/320);
    const auto* _39 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"tag_offset", /*type=*/_39,
                                    /*offset=*/384);
    const auto* _40 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_40, /*offset=*/448);
    const auto* _41 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] =
        PJCreateStructField(/*name=*/"size", /*type=*/_41, /*offset=*/512);
    const auto* _42 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_42, /*offset=*/576);
    const char* _43[3] = {"pj", "reflect", "InlineVariant"};
    const PJStructType* _44 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_43, /*type_domain=*/domain,
        /*num_fields=*/8, /*fields=*/fields, /*size=*/640, /*alignment=*/8);
    return _44;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::Sign> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[3];
    const PJUnitType* _45 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kSigned", /*type=*/_45, /*tag=*/1);
    const PJUnitType* _46 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kSignless", /*type=*/_46, /*tag=*/3);
    const PJUnitType* _47 = PJCreateUnitType(ctx);
    terms[2] = PJCreateTerm(/*name=*/"kUnsigned", /*type=*/_47, /*tag=*/2);
    const char* _48[2] = {"pj", "Sign"};
    const PJInlineVariantType* _49 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_48, /*type_domain=*/domain,
        /*num_terms=*/3, /*terms=*/terms, /*term_offset=*/0, /*term_size=*/0,
        /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8, /*alignment=*/8);
    return _49;
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
    const auto* _50 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[0] =
        PJCreateStructField(/*name=*/"width", /*type=*/_50, /*offset=*/0);
    const auto* _51 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[1] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_51, /*offset=*/64);
    const auto* _52 = BuildPJType<::v0_1::pj::Sign>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"sign", /*type=*/_52, /*offset=*/128);
    const char* _53[3] = {"pj", "reflect", "Int"};
    const PJStructType* _54 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_53, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/136, /*alignment=*/8);
    return _54;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::OutlineVariant> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[6];
    const auto* _57 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _56 = PJCreateVectorType(
        ctx, /*elem=*/_57, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _55 = PJCreateVectorType(
        ctx, /*elem=*/_56, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_55, /*offset=*/0);
    const auto* _59 =
        BuildPJType<::v0_1::pj::reflect::Term>::build(ctx, domain);
    const auto* _58 = PJCreateVectorType(
        ctx, /*elem=*/_59, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"terms", /*type=*/_58, /*offset=*/128);
    const auto* _60 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"tag_width", /*type=*/_60, /*offset=*/256);
    const auto* _61 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] = PJCreateStructField(/*name=*/"tag_alignment", /*type=*/_61,
                                    /*offset=*/320);
    const auto* _62 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"term_offset", /*type=*/_62,
                                    /*offset=*/384);
    const auto* _63 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"term_alignment", /*type=*/_63,
                                    /*offset=*/448);
    const char* _64[3] = {"pj", "reflect", "OutlineVariant"};
    const PJStructType* _65 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_64, /*type_domain=*/domain,
        /*num_fields=*/6, /*fields=*/fields, /*size=*/512, /*alignment=*/8);
    return _65;
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
    const auto* _66 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"type", /*type=*/_66, /*offset=*/0);
    const auto* _68 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _67 = PJCreateVectorType(
        ctx, /*elem=*/_68, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"name", /*type=*/_67, /*offset=*/32);
    const auto* _69 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"offset", /*type=*/_69, /*offset=*/160);
    const char* _70[3] = {"pj", "reflect", "StructField"};
    const PJStructType* _71 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_70, /*type_domain=*/domain,
        /*num_fields=*/3, /*fields=*/fields, /*size=*/224, /*alignment=*/8);
    return _71;
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
    const auto* _74 = PJCreateIntType(ctx, /*width=*/8, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNLESS);
    const auto* _73 = PJCreateVectorType(
        ctx, /*elem=*/_74, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    const auto* _72 = PJCreateVectorType(
        ctx, /*elem=*/_73, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[0] =
        PJCreateStructField(/*name=*/"name", /*type=*/_72, /*offset=*/0);
    const auto* _76 =
        BuildPJType<::v0_1::pj::reflect::StructField>::build(ctx, domain);
    const auto* _75 = PJCreateVectorType(
        ctx, /*elem=*/_76, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[1] =
        PJCreateStructField(/*name=*/"fields", /*type=*/_75, /*offset=*/128);
    const auto* _77 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] =
        PJCreateStructField(/*name=*/"size", /*type=*/_77, /*offset=*/256);
    const auto* _78 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[3] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_78, /*offset=*/320);
    const char* _79[3] = {"pj", "reflect", "Struct"};
    const PJStructType* _80 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_79, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/384, /*alignment=*/8);
    return _80;
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
    const char* _81[3] = {"pj", "reflect", "Unit"};
    const PJStructType* _82 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_81, /*type_domain=*/domain,
        /*num_fields=*/0, /*fields=*/fields, /*size=*/0, /*alignment=*/8);
    return _82;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::ReferenceMode> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[2];
    const PJUnitType* _83 = PJCreateUnitType(ctx);
    terms[0] = PJCreateTerm(/*name=*/"kOffset", /*type=*/_83, /*tag=*/2);
    const PJUnitType* _84 = PJCreateUnitType(ctx);
    terms[1] = PJCreateTerm(/*name=*/"kPointer", /*type=*/_84, /*tag=*/1);
    const char* _85[2] = {"pj", "ReferenceMode"};
    const PJInlineVariantType* _86 = PJCreateInlineVariantType(
        ctx, /*name_size=*/2, /*name=*/_85, /*type_domain=*/domain,
        /*num_terms=*/2, /*terms=*/terms, /*term_offset=*/0, /*term_size=*/0,
        /*tag_offset=*/0, /*tag_width=*/8, /*size=*/8, /*alignment=*/8);
    return _86;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Vector> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJStructField* fields[13];
    const auto* _87 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"elem", /*type=*/_87, /*offset=*/0);
    const auto* _88 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_UNSIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"min_length", /*type=*/_88, /*offset=*/32);
    const auto* _89 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[2] =
        PJCreateStructField(/*name=*/"max_length", /*type=*/_89, /*offset=*/96);
    const auto* _90 = PJCreateIntType(ctx, /*width=*/64, /*alignment=*/8,
                                      /*sign=*/PJ_SIGN_SIGNED);
    fields[3] =
        PJCreateStructField(/*name=*/"ppl_count", /*type=*/_90, /*offset=*/160);
    const auto* _91 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[4] = PJCreateStructField(/*name=*/"length_offset", /*type=*/_91,
                                    /*offset=*/224);
    const auto* _92 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[5] = PJCreateStructField(/*name=*/"length_size", /*type=*/_92,
                                    /*offset=*/288);
    const auto* _93 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[6] = PJCreateStructField(/*name=*/"ref_offset", /*type=*/_93,
                                    /*offset=*/352);
    const auto* _94 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[7] =
        PJCreateStructField(/*name=*/"ref_size", /*type=*/_94, /*offset=*/416);
    const auto* _95 =
        BuildPJType<::v0_1::pj::ReferenceMode>::build(ctx, domain);
    fields[8] = PJCreateStructField(/*name=*/"reference_mode", /*type=*/_95,
                                    /*offset=*/480);
    const auto* _96 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[9] = PJCreateStructField(/*name=*/"inline_payload_offset",
                                    /*type=*/_96, /*offset=*/488);
    const auto* _97 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[10] = PJCreateStructField(/*name=*/"partial_payload_offset",
                                     /*type=*/_97, /*offset=*/552);
    const auto* _98 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[11] =
        PJCreateStructField(/*name=*/"size", /*type=*/_98, /*offset=*/616);
    const auto* _99 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[12] =
        PJCreateStructField(/*name=*/"alignment", /*type=*/_99, /*offset=*/680);
    const char* _100[3] = {"pj", "reflect", "Vector"};
    const PJStructType* _101 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_100, /*type_domain=*/domain,
        /*num_fields=*/13, /*fields=*/fields, /*size=*/744, /*alignment=*/8);
    return _101;
  }
};
}  // namespace gen
}  // namespace pj

namespace pj {
namespace gen {
template <>
struct BuildPJType<::v0_1::pj::reflect::Type> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    const PJTerm* terms[7];
    const auto* _102 =
        BuildPJType<::v0_1::pj::reflect::Array>::build(ctx, domain);
    terms[0] = PJCreateTerm(/*name=*/"Array", /*type=*/_102, /*tag=*/6);
    const auto* _103 =
        BuildPJType<::v0_1::pj::reflect::InlineVariant>::build(ctx, domain);
    terms[1] = PJCreateTerm(/*name=*/"InlineVariant", /*type=*/_103, /*tag=*/4);
    const auto* _104 =
        BuildPJType<::v0_1::pj::reflect::Int>::build(ctx, domain);
    terms[2] = PJCreateTerm(/*name=*/"Int", /*type=*/_104, /*tag=*/1);
    const auto* _105 =
        BuildPJType<::v0_1::pj::reflect::OutlineVariant>::build(ctx, domain);
    terms[3] =
        PJCreateTerm(/*name=*/"OutlineVariant", /*type=*/_105, /*tag=*/5);
    const auto* _106 =
        BuildPJType<::v0_1::pj::reflect::Struct>::build(ctx, domain);
    terms[4] = PJCreateTerm(/*name=*/"Struct", /*type=*/_106, /*tag=*/3);
    const auto* _107 =
        BuildPJType<::v0_1::pj::reflect::Unit>::build(ctx, domain);
    terms[5] = PJCreateTerm(/*name=*/"Unit", /*type=*/_107, /*tag=*/2);
    const auto* _108 =
        BuildPJType<::v0_1::pj::reflect::Vector>::build(ctx, domain);
    terms[6] = PJCreateTerm(/*name=*/"Vector", /*type=*/_108, /*tag=*/7);
    const char* _109[3] = {"pj", "reflect", "Type"};
    const PJInlineVariantType* _110 = PJCreateInlineVariantType(
        ctx, /*name_size=*/3, /*name=*/_109, /*type_domain=*/domain,
        /*num_terms=*/7, /*terms=*/terms, /*term_offset=*/0, /*term_size=*/744,
        /*tag_offset=*/744, /*tag_width=*/8, /*size=*/752, /*alignment=*/8);
    return _110;
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
    const auto* _111 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[0] =
        PJCreateStructField(/*name=*/"pj_version", /*type=*/_111, /*offset=*/0);
    const auto* _112 = PJCreateIntType(ctx, /*width=*/32, /*alignment=*/8,
                                       /*sign=*/PJ_SIGN_SIGNED);
    fields[1] =
        PJCreateStructField(/*name=*/"head", /*type=*/_112, /*offset=*/32);
    const auto* _113 = BuildPJType<::v0_1::pj::Width>::build(ctx, domain);
    fields[2] = PJCreateStructField(/*name=*/"buffer_offset", /*type=*/_113,
                                    /*offset=*/64);
    const auto* _115 =
        BuildPJType<::v0_1::pj::reflect::Type>::build(ctx, domain);
    const auto* _114 = PJCreateVectorType(
        ctx, /*elem=*/_115, /*min_length=*/0, /*max_length=*/-1,
        /*wire_min_length=*/0, /*ppl_count=*/0, /*length_offset=*/0,
        /*length_size=*/64, /*ref_offset=*/64, /*ref_size=*/64,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/64, /*inline_payload_size=*/0,
        /*partial_payload_offset=*/128, /*partial_payload_size=*/0,
        /*size=*/128, /*alignment=*/8, /*outlined_payload_alignment=*/8);
    fields[3] =
        PJCreateStructField(/*name=*/"types", /*type=*/_114, /*offset=*/128);
    const char* _116[3] = {"pj", "reflect", "Protocol"};
    const PJStructType* _117 = PJCreateStructType(
        ctx, /*name_size=*/3, /*name=*/_116, /*type_domain=*/domain,
        /*num_fields=*/4, /*fields=*/fields, /*size=*/256, /*alignment=*/8);
    return _117;
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
    const auto* _118 =
        BuildPJType<::v0_1::pj::reflect::Protocol>::build(ctx, domain);
    return PJCreateProtocolType(ctx, _118, 0);
  }
};
}  // namespace gen

}  // namespace pj
