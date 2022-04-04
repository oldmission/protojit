#include "runtime.h"
#include "arch.hpp"
#include "context.hpp"
#include "defer.hpp"
#include "types.hpp"

#include <cstring>
#include <vector>

pj::types::Vector::ReferenceMode ConvertReferenceMode(
    PJReferenceMode reference_mode) {
  return (reference_mode == PJ_REFERENCE_MODE_POINTER)
             ? pj::types::Vector::kPointer
             : pj::types::Vector::kOffset;
}

pj::types::TypeDomain ConvertTypeDomain(PJTypeDomain type_domain) {
  return (type_domain == PJ_TYPE_DOMAIN_HOST) ? pj::types::TypeDomain::kHost
                                              : pj::types::TypeDomain::kWire;
}

pj::types::Int::Sign ConvertSign(PJSign sign) {
  switch (sign) {
    case PJ_SIGN_SIGNED:
      return pj::types::Int::kSigned;
    case PJ_SIGN_UNSIGNED:
      return pj::types::Int::kUnsigned;
    case PJ_SIGN_SIGNLESS:
      return pj::types::Int::kSignless;
    default:
      return pj::types::Int::kSignless;
  }
}

const PJUnitType* PJCreateUnitType(PJContext* c) {
  pj::ProtoJitContext* ctx = reinterpret_cast<pj::ProtoJitContext*>(c);

  auto unit_type = pj::types::StructType::get(
      &ctx->ctx_, pj::types::TypeDomain::kHost, "<unit>");
  unit_type.setTypeData(
      {.fields = llvm::ArrayRef<pj::types::StructField>{nullptr, 0ul},
       .size = pj::Bytes(0),
       .alignment = pj::Bytes(0)});
  return reinterpret_cast<const PJUnitType*>(unit_type.getAsOpaquePointer());
}

const PJIntType* PJCreateIntType(PJContext* c, Bits width, Bits alignment,
                                 PJSign sign) {
  auto int_type =
      pj::types::IntType::get(&reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_,
                              pj::types::Int{.width = pj::Bits(width),
                                             .alignment = pj::Bits(alignment),
                                             .sign = ConvertSign(sign)});
  return reinterpret_cast<const PJIntType*>(int_type.getAsOpaquePointer());
}

const PJStructField* PJCreateStructField(const char* name, const void* type,
                                         Bits offset) {
  const auto* field =
      new pj::types::StructField{.type = mlir::Type::getFromOpaquePointer(type),
                                 .name = name,
                                 .offset = pj::Bits(offset)};
  return reinterpret_cast<const PJStructField*>(field);
}

const PJStructType* PJCreateStructType(PJContext* c, uintptr_t name_size,
                                       const char* name[],
                                       PJTypeDomain type_domain,
                                       uintptr_t num_fields,
                                       const PJStructField* fields[], Bits size,
                                       Bits alignment) {
  pj::ProtoJitContext* ctx = reinterpret_cast<pj::ProtoJitContext*>(c);

  pj::types::ArrayRefConverter<llvm::StringRef> name_converter{name, name_size};
  pj::types::ArrayRefConverter<pj::types::StructField> fields_converter{
      fields, num_fields, [](const PJStructField* f) {
        auto casted = reinterpret_cast<const pj::types::StructField*>(f);
        DEFER(delete casted);
        return *casted;
      }};
  auto struct_type = pj::types::StructType::get(
      &ctx->ctx_, ConvertTypeDomain(type_domain), name_converter.get());
  struct_type.setTypeData({.fields = fields_converter.get(),
                           .size = pj::Bits(size),
                           .alignment = pj::Bits(alignment)});

  return reinterpret_cast<const PJStructType*>(
      struct_type.getAsOpaquePointer());
}

const PJTerm* PJCreateTerm(const char* name, const void* type, uint64_t tag) {
  const auto* term = new pj::types::Term{
      .name = name, .type = mlir::Type::getFromOpaquePointer(type), .tag = tag};
  return reinterpret_cast<const PJTerm*>(term);
}

const PJInlineVariantType* PJCreateInlineVariantType(
    PJContext* c, uintptr_t name_size, const char* name[],
    PJTypeDomain type_domain, uintptr_t num_terms, const PJTerm* terms[],
    Bits term_offset, Bits term_size, Bits tag_offset, Bits tag_width,
    Bits size, Bits alignment) {
  pj::ProtoJitContext* ctx = reinterpret_cast<pj::ProtoJitContext*>(c);

  pj::types::ArrayRefConverter<llvm::StringRef> name_converter{name, name_size};
  pj::types::ArrayRefConverter<pj::types::Term> terms_converter{
      terms, num_terms, [](const PJTerm* t) {
        auto casted = reinterpret_cast<const pj::types::Term*>(t);
        DEFER(delete casted);
        return *casted;
      }};
  auto inline_variant_type = pj::types::InlineVariantType::get(
      &ctx->ctx_, ConvertTypeDomain(type_domain), name_converter.get());
  inline_variant_type.setTypeData({.terms = terms_converter.get(),
                                   .term_offset = pj::Bits(term_offset),
                                   .term_size = pj::Bits(term_size),
                                   .tag_offset = pj::Bits(tag_offset),
                                   .tag_width = pj::Bits(tag_width),
                                   .size = pj::Bits(size),
                                   .alignment = pj::Bits(alignment)});

  return reinterpret_cast<const PJInlineVariantType*>(
      inline_variant_type.getAsOpaquePointer());
}

const PJOutlineVariantType* PJCreateOutlineVariantType(
    PJContext* c, uintptr_t name_size, const char* name[],
    PJTypeDomain type_domain, uintptr_t num_terms, const PJTerm* terms[],
    Bits tag_width, Bits tag_alignment, Bits term_offset) {
  pj::ProtoJitContext* ctx = reinterpret_cast<pj::ProtoJitContext*>(c);

  pj::types::ArrayRefConverter<llvm::StringRef> name_converter{name, name_size};
  pj::types::ArrayRefConverter<pj::types::Term> terms_converter{
      terms, num_terms, [](const PJTerm* t) {
        auto casted = reinterpret_cast<const pj::types::Term*>(t);
        DEFER(delete casted);
        return *casted;
      }};
  auto outline_variant_type = pj::types::OutlineVariantType::get(
      &ctx->ctx_, ConvertTypeDomain(type_domain), name_converter.get());
  outline_variant_type.setTypeData({.terms = terms_converter.get(),
                                    .tag_width = pj::Bits(tag_width),
                                    .tag_alignment = pj::Bits(tag_alignment),
                                    .term_offset = pj::Bits(term_offset)});

  return reinterpret_cast<const PJOutlineVariantType*>(
      outline_variant_type.getAsOpaquePointer());
}

const PJArrayType* PJCreateArrayType(PJContext* c, const void* type,
                                     intptr_t length, Bits elem_size,
                                     Bits alignment) {
  auto array_type = pj::types::ArrayType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_,
      pj::types::Array{.elem = mlir::Type::getFromOpaquePointer(type),
                       .length = length,
                       .elem_size = pj::Bits(elem_size),
                       .alignment = pj::Bits(alignment)});
  return reinterpret_cast<const PJArrayType*>(array_type.getAsOpaquePointer());
}

const PJVectorType* PJCreateVectorType(
    PJContext* c, const void* type, intptr_t min_length, intptr_t max_length,
    intptr_t ppl_count, Bits length_offset, Bits length_size, Bits ref_offset,
    Bits ref_size, PJReferenceMode reference_mode, Bits inline_payload_offset,
    Bits inline_payload_size, Bits partial_payload_offset,
    Bits partial_payload_size, Bits size, Bits alignment,
    Bits outlined_payload_alignment) {
  auto vector_type = pj::types::VectorType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_,
      pj::types::Vector{
          .elem = mlir::Type::getFromOpaquePointer(type),
          .min_length = min_length,
          .max_length = max_length,
          .ppl_count = ppl_count,
          .length_offset = pj::Bits(length_offset),
          .length_size = pj::Bits(length_size),
          .ref_offset = pj::Bits(ref_offset),
          .ref_size = pj::Bits(ref_size),
          .reference_mode = ConvertReferenceMode(reference_mode),
          .inline_payload_offset = pj::Bits(inline_payload_offset),
          .inline_payload_size = pj::Bits(inline_payload_size),
          .partial_payload_offset = pj::Bits(partial_payload_offset),
          .partial_payload_size = pj::Bits(partial_payload_size),
          .size = pj::Bits(size),
          .alignment = pj::Bits(alignment),
          .outlined_payload_alignment = pj::Bits(outlined_payload_alignment)});
  return reinterpret_cast<const PJVectorType*>(
      vector_type.getAsOpaquePointer());
}

const PJAnyType* PJCreateAnyType(PJContext* c, Bits data_ref_width,
                                 Bits data_ref_offset, Bits type_ref_width,
                                 Bits type_ref_offset, Bits tag_width,
                                 Bits tag_offset, Bits version_width,
                                 Bits version_offset, Bits size,
                                 Bits alignment) {
  auto any_type = pj::types::AnyType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_,
      pj::types::Any{.data_ref_width = pj::Bits(data_ref_width),
                     .data_ref_offset = pj::Bits(data_ref_offset),
                     .type_ref_width = pj::Bits(type_ref_width),
                     .type_ref_offset = pj::Bits(type_ref_offset),
                     .tag_width = pj::Bits(tag_width),
                     .tag_offset = pj::Bits(tag_offset),
                     .version_width = pj::Bits(version_width),
                     .version_offset = pj::Bits(version_offset),
                     .size = pj::Bits(size),
                     .alignment = pj::Bits(alignment)});
  return reinterpret_cast<const PJAnyType*>(any_type.getAsOpaquePointer());
}
