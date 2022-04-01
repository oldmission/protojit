#include "runtime.h"
#include "arch.hpp"
#include "scope.hpp"
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

llvm::ArrayRef<llvm::StringRef> ConvertStringArray(pj::Scope* S, uintptr_t size,
                                                   const char* strings[]) {
  llvm::StringRef* storage = reinterpret_cast<llvm::StringRef*>(
      S->Allocate(size * sizeof(llvm::StringRef)));

  for (uintptr_t i = 0; i < size; ++i) {
    uintptr_t len = std::strlen(strings[i]);
    char* buffer = reinterpret_cast<char*>(S->Allocate(len * sizeof(char)));
    std::memcpy(buffer, strings[i], len);
    storage[i] = llvm::StringRef{buffer, len};
  }

  return llvm::ArrayRef<llvm::StringRef>{storage, size};
}

const PJUnitType* PJCreateUnitType(PJContext* c) {
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);
  return reinterpret_cast<const PJUnitType*>(S->Unit().getAsOpaquePointer());
}

const PJIntType* PJCreateIntType(PJContext* c, Bits width, Bits alignment,
                                 PJSign sign) {
  auto int_type =
      pj::types::IntType::get(reinterpret_cast<pj::Scope*>(c)->Context(),
                              pj::types::Int{.width = pj::Bits(width),
                                             .alignment = pj::Bits(alignment),
                                             .sign = ConvertSign(sign)});
  return reinterpret_cast<const PJIntType*>(int_type.getAsOpaquePointer());
}

const PJStructField* PJCreateStructField(PJContext* c, const char* name,
                                         const void* type, Bits offset) {
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);
  const auto* field = new (S)
      pj::types::StructField{.type = mlir::Type::getFromOpaquePointer(type),
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
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);

  std::vector<pj::types::StructField> storage;
  for (uintptr_t i = 0; i < num_fields; ++i) {
    storage.push_back(
        *reinterpret_cast<const pj::types::StructField*>(fields[i]));
  }

  auto struct_type = pj::types::StructType::get(
      S->Context(), ConvertTypeDomain(type_domain),
      ConvertStringArray(S, name_size, name),
      pj::types::Struct{
          .fields =
              llvm::ArrayRef<pj::types::StructField>{&storage[0], num_fields},
          .size = pj::Bits(size),
          .alignment = pj::Bits(alignment)});

  return reinterpret_cast<const PJStructType*>(
      struct_type.getAsOpaquePointer());
}

const PJTerm* PJCreateTerm(PJContext* c, const char* name, const void* type,
                           uint64_t tag) {
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);
  const auto* term = new (S) pj::types::Term{
      .name = name, .type = mlir::Type::getFromOpaquePointer(type), .tag = tag};
  return reinterpret_cast<const PJTerm*>(term);
}

const PJInlineVariantType* PJCreateInlineVariantType(
    PJContext* c, uintptr_t name_size, const char* name[],
    PJTypeDomain type_domain, uintptr_t num_terms, const PJTerm* terms[],
    Bits term_offset, Bits term_size, Bits tag_offset, Bits tag_width,
    Bits size, Bits alignment) {
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);

  std::vector<pj::types::Term> storage;
  for (uintptr_t i = 0; i < num_terms; ++i) {
    storage.push_back(*reinterpret_cast<const pj::types::Term*>(terms[i]));
  }

  auto inline_variant_type = pj::types::InlineVariantType::get(
      S->Context(), ConvertTypeDomain(type_domain),
      ConvertStringArray(S, name_size, name),
      pj::types::InlineVariant{
          .terms = llvm::ArrayRef<pj::types::Term>{&storage[0], num_terms},
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
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);

  std::vector<pj::types::Term> storage;
  for (uintptr_t i = 0; i < num_terms; ++i) {
    storage.push_back(*reinterpret_cast<const pj::types::Term*>(terms[i]));
  }

  auto outline_variant_type = pj::types::OutlineVariantType::get(
      S->Context(), ConvertTypeDomain(type_domain),
      ConvertStringArray(S, name_size, name),
      pj::types::OutlineVariant{
          .terms = llvm::ArrayRef<pj::types::Term>{&storage[0], num_terms},
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
      reinterpret_cast<pj::Scope*>(c)->Context(),
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
      reinterpret_cast<pj::Scope*>(c)->Context(),
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
      reinterpret_cast<pj::Scope*>(c)->Context(),
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

const PJProtocolType* PJCreateProtocolType(PJContext* c,
                                           const void* head_type) {
  pj::Scope* S = reinterpret_cast<pj::Scope*>(c);
  pj::types::ValueType head = mlir::Type::getFromOpaquePointer(head_type)
                                  .dyn_cast<pj::types::ValueType>();
  if (!head) {
    return nullptr;
  }
  auto protocol_type = pj::types::ProtocolType::get(
      S->Context(), pj::types::Protocol{.head = head});
  return reinterpret_cast<const PJProtocolType*>(
      protocol_type.getAsOpaquePointer());
}

const PJRawBufferType* PJCreateRawBufferType(PJContext* c) {
  auto buffer_type =
      pj::types::RawBufferType::get(reinterpret_cast<pj::Scope*>(c)->Context());
  return reinterpret_cast<const PJRawBufferType*>(
      buffer_type.getAsOpaquePointer());
}

const PJBoundedBufferType* PJCreateBoundedBufferType(PJContext* c) {
  auto buffer_type = pj::types::BoundedBufferType::get(
      reinterpret_cast<pj::Scope*>(c)->Context());
  return reinterpret_cast<const PJBoundedBufferType*>(
      buffer_type.getAsOpaquePointer());
}
