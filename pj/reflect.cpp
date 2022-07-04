#include "pj/reflect.hpp"
#include "pj/array_ref.hpp"
#include "pj/protojit.hpp"

namespace pj {

namespace reflect {

#define FOR_EACH_REFLECTABLE_PROTOJIT_TYPE(V) \
  V(Int)                                      \
  V(Unit)                                     \
  V(Struct)                                   \
  V(InlineVariant)                            \
  V(OutlineVariant)                           \
  V(Array)                                    \
  V(Vector)

#define DECLARE_TYPE_REFLECTORS(T)                                         \
  void reflect(::pj::types::T##Type type, llvm::BumpPtrAllocator& alloc,   \
               std::vector<Type>& pool,                                    \
               std::unordered_map<const void*, int32_t>& cache);           \
  ::pj::types::ValueType unreflect(                                        \
      const ::pj::reflect::T& type, int32_t index, mlir::MLIRContext& ctx, \
      types::WireDomainAttr domain, ArrayRef<Type> pool);
FOR_EACH_REFLECTABLE_PROTOJIT_TYPE(DECLARE_TYPE_REFLECTORS)
#undef DECLARE_TYPE_REFLECTORS

int32_t reflect(types::ValueType type, llvm::BumpPtrAllocator& alloc,
                std::vector<Type>& pool,
                std::unordered_map<const void*, int32_t>& cache) {
  if (auto it = cache.find(type.getAsOpaquePointer()); it != cache.end()) {
    return it->second;
  }

#define MATCH_TYPE(T)                                        \
  else if (auto t = type.dyn_cast<::pj::types::T##Type>()) { \
    reflect(t, alloc, pool, cache);                          \
  }

  if (false)
    ;
  FOR_EACH_REFLECTABLE_PROTOJIT_TYPE(MATCH_TYPE)
#undef MATCH_TYPE
  else {
    UNREACHABLE();
  }

  auto result = pool.size() - 1;
  cache.emplace(type.getAsOpaquePointer(), result);
  return result;
}

types::ValueType unreflect(const Type& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
#define MATCH_TYPE(T)                                         \
  else if (type.tag == Type::Kind::T) {                       \
    return unreflect(type.value.T, index, ctx, domain, pool); \
  }

  if (false)
    ;
  FOR_EACH_REFLECTABLE_PROTOJIT_TYPE(MATCH_TYPE)
#undef MATCH_TYPE
  else {
    UNREACHABLE();
  }
}

static constexpr uint32_t kProtojitMajorVersion = 0;

Protocol reflect(llvm::BumpPtrAllocator& alloc, types::ProtocolType protocol) {
  std::vector<Type> pool;
  std::unordered_map<const void*, int32_t> cache;
  const int32_t head_offset = reflect(protocol->head, alloc, pool, cache);
  auto* pool_alloc = alloc.Allocate<Type>(pool.size());
  std::copy(pool.begin(), pool.end(), pool_alloc);

  const auto proto = Protocol{
      .pj_version = kProtojitMajorVersion,
      .head = head_offset,
      .buffer_offset = protocol->buffer_offset,
      .types = {pool_alloc, pool.size()},
  };

  return proto;
}

types::ValueType unreflect(const Protocol& type, mlir::MLIRContext& ctx,
                           types::WireDomainAttr domain) {
  ArrayRef<Type> pool{type.types.begin(), type.types.size()};
  const Type& head = pool[type.head];
  return types::ProtocolType::get(
      &ctx, types::Protocol{
                .head = unreflect(head, type.head, ctx, domain, pool),
                .buffer_offset = type.buffer_offset,
            });
}

void reflect(types::IntType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  Type result(Type::Int{},  //
              Int{
                  .width = type->width,
                  .alignment = type->alignment,
                  .sign = type->sign,
              });
  pool.emplace_back(result);
}

types::ValueType unreflect(const Int& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  return types::IntType::get(&ctx, types::Int{
                                       .width = type.width,
                                       .alignment = type.alignment,
                                       .sign = type.sign,
                                   });
}

void reflect(types::UnitType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  pool.emplace_back(Type::Unit{});
}

types::ValueType unreflect(const Unit& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  return types::UnitType::get(&ctx);
}

Name reflectName(types::Name name, llvm::BumpPtrAllocator& alloc) {
  auto* result = alloc.Allocate<span<pj_char>>(name.size());
  for (size_t i = 0; i < name.size(); ++i) {
    result[i] = {name[i].data(), name[i].size()};
  }
  return {result, name.size()};
}

ArrayRefConverter<llvm::StringRef> unreflectName(Name name) {
  return {name.begin(), name.size(), [](auto str) {
            return llvm::StringRef{str.begin(), str.size()};
          }};
}

void reflect(types::StructType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  auto* fields = alloc.Allocate<StructField>(type->fields.size());
  auto* fields_it = fields;
  for (auto& field : type->fields) {
    (*fields_it++) = StructField{
        .type = reflect(field.type, alloc, pool, cache),
        .name = {field.name.data(), field.name.size()},
        .offset = field.offset,
    };
  }
  const int32_t this_offset = pool.size();
  for (size_t i = 0; i < type->fields.size(); ++i) {
    fields[i].type = fields[i].type - this_offset;
  }
  pool.emplace_back(   //
      Type::Struct{},  //
      Struct{
          .name = reflectName(type.name(), alloc),
          .fields = {fields, type->fields.size()},
          .size = type->size,
          .alignment = type->alignment,
      });
}

types::ValueType unreflect(const Struct& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  auto name_conv = unreflectName(type.name);
  ArrayRefConverter<types::StructField> field_conv{
      type.fields, type.fields.size(), [&](const StructField& f) {
        return types::StructField{
            .type = unreflect(pool[index + f.type], index + f.type, ctx, domain,
                              pool),
            .name = {f.name.begin(), f.name.size()},
            .offset = f.offset,
        };
      }};
  auto result = types::StructType::get(&ctx, domain, name_conv.get());
  result.setTypeData(types::Struct{
      .fields = field_conv.get(),
      .size = type.size,
      .alignment = type.alignment,
  });
  return result;
}

void reflect(types::ArrayType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  const int32_t elem = reflect(type->elem, alloc, pool, cache);
  const int32_t elem_offset = elem - pool.size();
  pool.emplace_back(  //
      Type::Array{},  //
      Array{
          .elem = elem_offset,
          .length = type->length,
          .elem_size = type->elem_size,
          .alignment = type->alignment,
      });
}

types::ValueType unreflect(const Array& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  return types::ArrayType::get(
      &ctx, types::Array{
                .elem = unreflect(pool[index + type.elem], index + type.elem,
                                  ctx, domain, pool),
                .length = type.length,
                .elem_size = type.elem_size,
                .alignment = type.alignment,
            });
}

void reflect(types::VectorType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  const int32_t elem = reflect(type->elem, alloc, pool, cache);
  const int32_t elem_offset = elem - pool.size();
  pool.emplace_back(  //
      Type::Vector{},
      Vector{
          .elem = elem_offset,
          .min_length = type->min_length,
          .max_length = type->max_length,
          .ppl_count = type->ppl_count,
          .length_offset = type->length_offset,
          .length_size = type->length_size,
          .ref_offset = type->ref_offset,
          .ref_size = type->ref_size,
          .reference_mode = type->reference_mode,
          .inline_payload_offset = type->inline_payload_offset,
          .partial_payload_offset = type->partial_payload_offset,
          .size = type->size,
          .alignment = type->alignment,
      });
}

types::ValueType unreflect(const Vector& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  return types::VectorType::get(
      &ctx, types::Vector{
                .elem = unreflect(pool[index + type.elem], index + type.elem,
                                  ctx, domain, pool),
                .min_length = type.min_length,
                .max_length = type.max_length,
                .ppl_count = type.ppl_count,
                .length_offset = type.length_offset,
                .length_size = type.length_size,
                .ref_offset = type.ref_offset,
                .ref_size = type.ref_size,
                .reference_mode = type.reference_mode,
                .inline_payload_offset = type.inline_payload_offset,
                .partial_payload_offset = type.partial_payload_offset,
                .size = type.size,
                .alignment = type.alignment,
            });
}

Path reflectPath(types::PathAttr path, llvm::BumpPtrAllocator& alloc) {
  auto vec = path.getValue();
  auto* result = alloc.Allocate<span<pj_char>>(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = {vec[i].data(), vec[i].size()};
  }
  return {result, vec.size()};
}

types::PathAttr unreflectPath(Path path, mlir::MLIRContext& ctx) {
  ArrayRefConverter<llvm::StringRef> conv{
      path.begin(), path.size(), [](auto str) {
        return llvm::StringRef{str.begin(), str.size()};
      }};
  return types::PathAttr::get(&ctx, conv.get());
}

TermAttribute* reflectTermAttributes(ArrayRef<types::TermAttribute> attrs,
                                     llvm::BumpPtrAllocator& alloc) {
  auto* attrs_alloc = alloc.Allocate<TermAttribute>(attrs.size());
  auto* attrs_it = attrs_alloc;
  for (auto& attr : attrs) {
    (*attrs_it++) = std::visit(
        overloaded{[&](const types::TermAttribute::Undef& undef) {
                     return TermAttribute{};
                   },
                   [&](const types::TermAttribute::VectorSplit& vs) {
                     return TermAttribute{
                         TermAttribute::vector_split{},
                         VectorSplit{
                             .type = vs.type == types::TermAttribute::
                                                    VectorSplit::Type::kInline
                                         ? VectorSplitType::kInline
                                         : VectorSplitType::kOutline,
                             .inline_length = vs.inline_length,
                             .path = reflectPath(vs.path, alloc),
                             .is_default = vs.is_default,
                         }};
                   }},
        attr.value);
  }
  return attrs_alloc;
}

ArrayRef<types::TermAttribute> unreflectTermAttributes(
    span<TermAttribute> attrs, mlir::MLIRContext& ctx,
    llvm::BumpPtrAllocator& alloc) {
  auto* attrs_alloc = alloc.Allocate<types::TermAttribute>(attrs.size());
  auto* attrs_it = attrs_alloc;
  for (const TermAttribute& attr : attrs) {
    switch (attr.tag) {
      case TermAttribute::Kind::undef:
        // TODO: decode is_default if available using reflection.
        (*attrs_it++) = types::TermAttribute{
            .value = types::TermAttribute::Undef{.is_default = false},
        };
        break;
      case TermAttribute::Kind::vector_split:
        using VectorSplit = types::TermAttribute::VectorSplit;
        const auto& vs = attr.value.vector_split;
        (*attrs_it++) = types::TermAttribute{
            .value =
                types::TermAttribute::VectorSplit{
                    .type = vs.type == VectorSplitType::kInline
                                ? VectorSplit::Type::kInline
                                : VectorSplit::Type::kOutline,
                    .inline_length = vs.inline_length,
                    .path = unreflectPath(vs.path, ctx),
                    .is_default = !!vs.is_default,
                },
        };
        break;
    }
  }
  return {attrs_alloc, attrs.size()};
}

Term* reflectTerms(ArrayRef<types::Term> terms, llvm::BumpPtrAllocator& alloc,
                   std::vector<Type>& pool,
                   std::unordered_map<const void*, int32_t>& cache) {
  auto* terms_alloc = alloc.Allocate<Term>(terms.size());
  auto* terms_it = terms_alloc;
  for (auto& term : terms) {
    auto* attributes = reflectTermAttributes(term.attributes, alloc);
    (*terms_it++) = Term{
        .name = {term.name.data(), term.name.size()},
        .type = reflect(term.type, alloc, pool, cache),
        .tag = term.tag,
        .attributes = {attributes, term.attributes.size()},
    };
  }
  const int32_t this_offset = pool.size();
  for (size_t i = 0; i < terms.size(); ++i) {
    terms_alloc[i].type = terms_alloc[i].type - this_offset;
  }
  return terms_alloc;
}

ArrayRef<types::Term> unreflectTerms(span<Term> terms, int32_t index,
                                     mlir::MLIRContext& ctx,
                                     llvm::BumpPtrAllocator& alloc,
                                     types::WireDomainAttr domain,
                                     ArrayRef<Type> pool) {
  auto* terms_alloc = alloc.Allocate<types::Term>(terms.size());
  auto* terms_it = terms_alloc;
  for (const Term& term : terms) {
    (*terms_it++) = types::Term{
        .name = {term.name.begin(), term.name.size()},
        .type = unreflect(pool[index + term.type], index + term.type, ctx,
                          domain, pool),
        .tag = term.tag,
        .attributes = unreflectTermAttributes(term.attributes, ctx, alloc),
    };
  }
  return {terms_alloc, terms.size()};
}

void reflect(types::InlineVariantType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  auto* terms = reflectTerms(type->terms, alloc, pool, cache);
  pool.emplace_back(          //
      Type::InlineVariant(),  //
      InlineVariant{
          .name = reflectName(type.name(), alloc),
          .terms = {terms, type->terms.size()},
          .term_offset = type->term_offset,
          .term_size = type->term_size,
          .tag_offset = type->tag_offset,
          .tag_width = type->tag_width,
          .size = type->size,
          .alignment = type->alignment,
      });
}

types::ValueType unreflect(const InlineVariant& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  llvm::BumpPtrAllocator alloc;
  auto name_conv = unreflectName(type.name);
  auto result = types::InlineVariantType::get(&ctx, domain, name_conv.get());
  result.setTypeData(types::InlineVariant{
      .terms = unreflectTerms(type.terms, index, ctx, alloc, domain, pool),
      .term_offset = type.term_offset,
      .term_size = type.term_size,
      .tag_offset = type.tag_offset,
      .tag_width = type.tag_width,
      .size = type.size,
      .alignment = type.alignment,
  });
  return result;
}

void reflect(types::OutlineVariantType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  auto* terms = reflectTerms(type->terms, alloc, pool, cache);
  pool.emplace_back(           //
      Type::OutlineVariant{},  //
      OutlineVariant{
          .name = reflectName(type.name(), alloc),
          .terms = {terms, type->terms.size()},
          .tag_width = type->tag_width,
          .tag_alignment = type->tag_alignment,
          .term_offset = type->term_offset,
          .term_alignment = type->term_alignment,
      });
}

types::ValueType unreflect(const OutlineVariant& type, int32_t index,
                           mlir::MLIRContext& ctx, types::WireDomainAttr domain,
                           ArrayRef<Type> pool) {
  llvm::BumpPtrAllocator alloc;
  auto name_conv = unreflectName(type.name);
  auto result = types::OutlineVariantType::get(&ctx, domain, name_conv.get());
  result.setTypeData(types::OutlineVariant{
      .terms = unreflectTerms(type.terms, index, ctx, alloc, domain, pool),
      .tag_width = type.tag_width,
      .tag_alignment = type.tag_alignment,
      .term_offset = type.term_offset,
      .term_alignment = type.term_alignment,
  });
  return result;
}

#undef FOR_EACH_PROTOJIT_TYPE

types::ValueType reflectableTypeFor(types::ValueType type,
                                    types::ReflectDomainAttr domain) {
  if (type.isa<types::IntType>()) {
    return type;
  }
  if (auto array = type.dyn_cast<types::ArrayType>()) {
    types::Array ary{array};
    ary.elem = reflectableTypeFor(ary.elem, domain);
    return types::ArrayType::get(array.getContext(), ary);
  }
  if (auto vec = type.dyn_cast<types::VectorType>()) {
    auto elem = reflectableTypeFor(vec->elem, domain);
    return types::VectorType::get(
        type.getContext(),
        types::Vector{
            .elem = elem,
            .min_length = 0,
            .max_length = vec->max_length,
            .wire_min_length = 0,
            .ppl_count = 0,
            .length_offset = Bytes(0),
            .length_size = Bytes(8),
            .ref_offset = Bytes(8),
            .ref_size = Bytes(8),
            .reference_mode = ReferenceMode::kPointer,
            .inline_payload_offset = Width::None(),
            .inline_payload_size = Width::None(),
            .partial_payload_offset = Width::None(),
            .partial_payload_size = Width::None(),
            .size = Bytes(16),
            .alignment = Bytes(8),
            .outlined_payload_alignment = elem.headAlignment(),
        });
  }
  if (auto str = type.dyn_cast<types::StructType>()) {
    llvm::SmallVector<types::StructField, 4> fields;
    for (auto& f : str->fields) {
      fields.emplace_back(types::StructField{
          .type = reflectableTypeFor(f.type, domain),
          .name = f.name,
          .offset = f.offset,
      });
    }
    types::Struct result = str;
    result.fields = fields;
    auto typ = types::StructType::get(type.getContext(), domain, str.name());
    typ.setTypeData(result);
    return typ;
  }
  if (auto var = type.dyn_cast<types::VariantType>()) {
    llvm::SmallVector<types::Term, 4> terms;
    Width term_max_size = Bits(0);
    for (auto& t : var.terms()) {
      auto back = terms.emplace_back(types::Term{
          .name = t.name,
          .type = reflectableTypeFor(t.type, domain),
          .tag = t.tag,
      });
      term_max_size = std::max(term_max_size, back.type.headSize());
    }
    if (auto inl = type.dyn_cast<types::InlineVariantType>()) {
      types::InlineVariant result = inl;
      result.terms = terms;
      auto typ =
          types::InlineVariantType::get(type.getContext(), domain, inl.name());
      typ.setTypeData(result);
      return typ;
    }
    if (auto outl = type.dyn_cast<types::OutlineVariantType>()) {
      types::InlineVariant result{
          .terms = terms,
          .term_offset = Bytes(0),
          .term_size = term_max_size,
          .tag_offset = term_max_size,
          .tag_width = outl.tag_width(),
          .size = term_max_size + outl.tag_width(),
          .alignment = outl->term_alignment,
      };
      auto typ =
          types::InlineVariantType::get(type.getContext(), domain, outl.name());
      typ.setTypeData(result);
      return typ;
    }
  }
  UNREACHABLE();
}
}  // namespace reflect
}  // namespace pj
