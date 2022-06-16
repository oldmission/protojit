#include "runtime.h"
#include "arch.hpp"
#include "context.hpp"
#include "defer.hpp"
#include "plan.hpp"
#include "span.hpp"
#include "types.hpp"

#include <cstring>
#include <vector>

pj::ReferenceMode ConvertReferenceMode(PJReferenceMode reference_mode) {
  return (reference_mode == PJ_REFERENCE_MODE_POINTER)
             ? pj::ReferenceMode::kPointer
             : pj::ReferenceMode::kOffset;
}

pj::types::TypeDomain ConvertTypeDomain(PJTypeDomain type_domain) {
  return (type_domain == PJ_TYPE_DOMAIN_HOST) ? pj::types::TypeDomain::kHost
                                              : pj::types::TypeDomain::kWire;
}

pj::Sign ConvertSign(PJSign sign) {
  switch (sign) {
    case PJ_SIGN_SIGNED:
      return pj::Sign::kSigned;
    case PJ_SIGN_UNSIGNED:
      return pj::Sign::kUnsigned;
    case PJ_SIGN_SIGNLESS:
      return pj::Sign::kSignless;
    default:
      return pj::Sign::kSignless;
  }
}

pj::types::ProtocolType ConvertProtocol(const PJProtocol* p) {
  return mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(p))
      .cast<pj::types::ProtocolType>();
}

PJContext* PJGetContext() {
  return reinterpret_cast<PJContext*>(new pj::ProtoJitContext());
}

void PJFreeContext(PJContext* ctx) {
  delete reinterpret_cast<pj::ProtoJitContext*>(ctx);
}

const PJAnyType* PJCreateAnyType(PJContext* c, Bits data_ref_offset,
                                 Bits data_ref_width, Bits type_ref_offset,
                                 Bits type_ref_width, Bits size, Bits alignment,
                                 const void* self_type) {
  auto any = pj::types::Any{
      .data_ref_width = pj::Bits(data_ref_width),
      .data_ref_offset = pj::Bits(data_ref_offset),
      .type_ref_width = pj::Bits(type_ref_width),
      .type_ref_offset = pj::Bits(type_ref_offset),
      .size = pj::Bits(size),
      .alignment = pj::Bits(alignment),
      .self = mlir::Type::getFromOpaquePointer(self_type)
                  .cast<pj::types::ValueType>(),
  };
  auto any_type = pj::types::AnyType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_, any);
  return reinterpret_cast<const PJAnyType*>(any_type.getAsOpaquePointer());
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

const PJUnitType* PJCreateUnitType(PJContext* c) {
  auto unit_type = pj::types::UnitType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_);
  return reinterpret_cast<const PJUnitType*>(unit_type.getAsOpaquePointer());
}

const PJStructField* PJCreateStructField(const char* name, const void* type,
                                         Bits offset) {
  const auto* field = new pj::types::StructField{
      .type =
          mlir::Type::getFromOpaquePointer(type).cast<pj::types::ValueType>(),
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

  pj::SpanConverter<llvm::StringRef> name_converter{name, name_size};
  pj::SpanConverter<pj::types::StructField> fields_converter{
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
      .name = name,
      .type =
          mlir::Type::getFromOpaquePointer(type).cast<pj::types::ValueType>(),
      .tag = tag,
  };
  return reinterpret_cast<const PJTerm*>(term);
}

const PJInlineVariantType* PJCreateInlineVariantType(
    PJContext* c, uintptr_t name_size, const char* name[],
    PJTypeDomain type_domain, uintptr_t num_terms, const PJTerm* terms[],
    Bits term_offset, Bits term_size, Bits tag_offset, Bits tag_width,
    Bits size, Bits alignment) {
  pj::ProtoJitContext* ctx = reinterpret_cast<pj::ProtoJitContext*>(c);

  pj::SpanConverter<llvm::StringRef> name_converter{name, name_size};
  pj::SpanConverter<pj::types::Term> terms_converter{
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
    Bits tag_width, Bits tag_alignment, Bits term_offset, Bits term_alignment) {
  pj::ProtoJitContext* ctx = reinterpret_cast<pj::ProtoJitContext*>(c);

  pj::SpanConverter<llvm::StringRef> name_converter{name, name_size};
  pj::SpanConverter<pj::types::Term> terms_converter{
      terms, num_terms, [](const PJTerm* t) {
        auto casted = reinterpret_cast<const pj::types::Term*>(t);
        DEFER(delete casted);
        return *casted;
      }};
  auto outline_variant_type = pj::types::OutlineVariantType::get(
      &ctx->ctx_, ConvertTypeDomain(type_domain), name_converter.get());
  outline_variant_type.setTypeData(
      {.terms = terms_converter.get(),
       .tag_width = pj::Bits(tag_width),
       .tag_alignment = pj::Bits(tag_alignment),
       .term_offset = pj::Bits(term_offset),
       .term_alignment = pj::Bits(term_alignment)});

  return reinterpret_cast<const PJOutlineVariantType*>(
      outline_variant_type.getAsOpaquePointer());
}

const PJArrayType* PJCreateArrayType(PJContext* c, const void* type,
                                     uint64_t length, Bits elem_size,
                                     Bits alignment) {
  auto elem =
      mlir::Type::getFromOpaquePointer(type).cast<pj::types::ValueType>();
  auto array_type = pj::types::ArrayType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_,
      pj::types::Array{
          .elem = elem,
          .length = length,
          .elem_size = pj::Bits(elem_size),
          .alignment = pj::Bits(alignment),
      });
  return reinterpret_cast<const PJArrayType*>(array_type.getAsOpaquePointer());
}

const PJVectorType* PJCreateVectorType(
    PJContext* c, const void* type, intptr_t min_length, intptr_t max_length,
    intptr_t wire_min_length, intptr_t ppl_count, Bits length_offset,
    Bits length_size, Bits ref_offset, Bits ref_size,
    PJReferenceMode reference_mode, Bits inline_payload_offset,
    Bits inline_payload_size, Bits partial_payload_offset,
    Bits partial_payload_size, Bits size, Bits alignment,
    Bits outlined_payload_alignment) {
  auto vector_type = pj::types::VectorType::get(
      &reinterpret_cast<pj::ProtoJitContext*>(c)->ctx_,
      pj::types::Vector{
          .elem = mlir::Type::getFromOpaquePointer(type)
                      .cast<pj::types::ValueType>(),
          .min_length = min_length,
          .max_length = max_length,
          .wire_min_length = wire_min_length,
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

const PJProtocol* PJCreateProtocolType(PJContext* ctx_, const void* head_,
                                       Bits buffer_offset) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  auto head = mlir::Type::getFromOpaquePointer(head_);

  auto proto = pj::types::ProtocolType::get(
      &ctx->ctx_,
      pj::types::Protocol{.head = head.cast<pj::types::ValueType>(),
                          .buffer_offset = pj::Bits(buffer_offset)});
  return reinterpret_cast<const PJProtocol*>(proto.getAsOpaquePointer());
}

const PJProtocol* PJPlanProtocol(PJContext* ctx_, const void* head_,
                                 const char* tag_path_) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  auto head = mlir::Type::getFromOpaquePointer(head_);

  auto tag_path = pj::types::PathAttr::fromString(&ctx->ctx_, tag_path_);

  return reinterpret_cast<const PJProtocol*>(
      pj::plan_protocol(ctx->ctx_, head, tag_path).getAsOpaquePointer());
}

uint64_t PJGetProtoSize(PJContext* ctx_, const PJProtocol* proto) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  return ctx->getProtoSize(ConvertProtocol(proto));
}

void PJEncodeProto(PJContext* ctx_, const PJProtocol* proto, char* buf) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  ctx->encodeProto(ConvertProtocol(proto), buf);
}

const PJProtocol* PJDecodeProto(PJContext* ctx_, const char* buf) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  return reinterpret_cast<const PJProtocol*>(
      ctx->decodeProto(buf).getAsOpaquePointer());
}

void PJAddEncodeFunction(PJContext* ctx_, const char* name, const void* src_,
                         const PJProtocol* protocol_, const char* src_path) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  auto src = mlir::Type::getFromOpaquePointer(src_);
  auto protocol = ConvertProtocol(protocol_);
  ctx->addEncodeFunction(name, src, protocol, src_path);
}

void PJAddDecodeFunction(PJContext* ctx_, const char* name,
                         const PJProtocol* protocol_, const void* dest_,
                         uintptr_t num_handlers, const char* handlers_[]) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  auto protocol = ConvertProtocol(protocol_);
  auto dest = mlir::Type::getFromOpaquePointer(dest_);

  std::vector<std::string> handlers;
  for (uintptr_t i = 0; i < num_handlers; ++i) {
    handlers.push_back(handlers_[i]);
  }

  ctx->addDecodeFunction(name, protocol, dest, handlers);
}

void PJAddSizeFunction(PJContext* ctx_, const char* name, const void* src_,
                       const PJProtocol* protocol_, const char* src_path,
                       bool round_up) {
  auto* ctx = reinterpret_cast<pj::ProtoJitContext*>(ctx_);
  auto src = mlir::Type::getFromOpaquePointer(src_);
  auto protocol = ConvertProtocol(protocol_);
  ctx->addSizeFunction(name, src, protocol, src_path, round_up);
}

void PJPrecompile(PJContext* ctx, const char* filename) {
  reinterpret_cast<pj::ProtoJitContext*>(ctx)->precompile(filename);
}

const PJPortal* PJCompile(PJContext* ctx) {
  auto portal = reinterpret_cast<pj::ProtoJitContext*>(ctx)->compile();
  auto* handle = reinterpret_cast<const PJPortal*>(portal.release());
  return handle;
}

SizeFunction PJGetSizeFunction(const PJPortal* portal, const char* name) {
  return reinterpret_cast<const pj::Portal*>(portal)->GetSizeFunction<void>(
      name);
}

EncodeFunction PJGetEncodeFunction(const PJPortal* portal, const char* name) {
  return reinterpret_cast<const pj::Portal*>(portal)->GetEncodeFunction<void>(
      name);
}

DecodeFunction PJGetDecodeFunction(const PJPortal* portal, const char* name) {
  return reinterpret_cast<const pj::Portal*>(portal)
      ->GetDecodeFunction<void, BoundedBuffer>(name);
}

void PJFreePortal(const PJPortal* portal) {
  delete reinterpret_cast<const pj::Portal*>(portal);
}

namespace pj {

PJContext* getContext() { return PJGetContext(); }

void freeContext(PJContext* ctx) { PJFreeContext(ctx); }

uint64_t getProtoSize(PJContext* ctx, const PJProtocol* proto) {
  return PJGetProtoSize(ctx, proto);
}

void encodeProto(PJContext* ctx, const PJProtocol* proto, char* buf) {
  PJEncodeProto(ctx, proto, buf);
}

const PJProtocol* decodeProto(PJContext* ctx, const char* buf) {
  return PJDecodeProto(ctx, buf);
}

void precompile(PJContext* ctx, const std::string& filename) {
  return PJPrecompile(ctx, filename.c_str());
}

std::unique_ptr<Portal> compile(PJContext* ctx) {
  return reinterpret_cast<ProtoJitContext*>(ctx)->compile();
}

}  // namespace pj
