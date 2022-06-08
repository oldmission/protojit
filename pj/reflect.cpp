#include "pj/reflect.hpp"
#include "pj/protojit.hpp"

namespace pj {
using namespace types;

namespace reflect {

#define FOR_EACH_REFLECTABLE_PROTOJIT_TYPE(V) \
  V(Int)                                      \
  V(Struct)                                   \
  V(InlineVariant)                            \
  V(Array)                                    \
  V(Vector)

#define DECLARE_TYPE_REFLECTER(T)                                        \
  void reflect(::pj::types::T##Type type, llvm::BumpPtrAllocator& alloc, \
               std::vector<Type>& pool,                                  \
               std::unordered_map<const void*, int32_t>& cache);
FOR_EACH_REFLECTABLE_PROTOJIT_TYPE(DECLARE_TYPE_REFLECTER)
#undef DECLARE_TYPE_REFLECTER

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

static constexpr uint32_t kProtojitMajorVersion = 0;

Proto reflect(llvm::BumpPtrAllocator& alloc, types::ProtocolType protocol) {
  std::vector<Type> pool;
  std::unordered_map<const void*, int32_t> cache;
  const int32_t head_offset = reflect(protocol->head, alloc, pool, cache);
  auto* pool_alloc = alloc.Allocate<Type>(pool.size());
  std::copy(pool.begin(), pool.end(), pool_alloc);

  const auto proto = Proto{
      .head = head_offset,
      .pj_ver = kProtojitMajorVersion,
      .types = {pool_alloc, pool.size()},
  };

  return proto;
}

void reflect(::pj::types::IntType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  Type result{.tag = Type::Kind::Int};
  result.value.Int = Int{
      .width = type->width,
      .sign = type->sign,
  };
  pool.emplace_back(result);
}

Name reflectName(types::Name name, llvm::BumpPtrAllocator& alloc) {
  auto* result = alloc.Allocate<pj::ArrayView<char, 0, -1>>(name.size());
  for (size_t i = 0; i < name.size(); ++i) {
    result[i] = {name[i].data(), name[i].size()};
  }
  return {result, name.size()};
}

void reflect(StructType type, llvm::BumpPtrAllocator& alloc,
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
  Type typ{.tag = Type::Kind::Struct};
  typ.value.Struct = {
      .name = reflectName(type.name(), alloc),
      .fields = {fields, type->fields.size()},
      .size = type->size,
  };
  pool.emplace_back(typ);
}

void reflect(ArrayType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  const int32_t elem_offset =
      pool.size() - reflect(type->elem, alloc, pool, cache);
  Type typ{.tag = Type::Kind::Array};
  typ.value.Array = {
      .elem = elem_offset,
      .length = type->length,
      .elem_size = type->elem_size,
  };
  pool.emplace_back(typ);
}

void reflect(VectorType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  ASSERT(type->min_length == 0);
  ASSERT(type->ppl_count == 0);
  ASSERT(type->reference_mode == ReferenceMode::kPointer);
  const int32_t elem_offset =
      pool.size() - reflect(type->elem, alloc, pool, cache);
  Type typ{.tag = Type::Kind::Vector};
  typ.value.Vector = {
      .elem = elem_offset,
      .max_length = type->max_length,
      .length_offset = type->length_offset,
      .length_size = type->length_size,
      .ref_offset = type->ref_offset,
      .ref_size = type->ref_size,
  };
  pool.emplace_back(typ);
}

Term* reflectTerms(pj::Span<pj::types::Term> terms,
                   llvm::BumpPtrAllocator& alloc, std::vector<Type>& pool,
                   std::unordered_map<const void*, int32_t>& cache) {
  auto* terms_alloc = alloc.Allocate<Term>(terms.size());
  auto* terms_it = terms_alloc;
  for (auto& term : terms) {
    (*terms_it++) = Term{
        .name = {term.name.data(), term.name.size()},
        .type = reflect(term.type, alloc, pool, cache),
        .tag = term.tag,
    };
  }
  const int32_t this_offset = pool.size();
  for (size_t i = 0; i < terms.size(); ++i) {
    terms_alloc[i].type = terms_alloc[i].type - this_offset;
  }
  return terms_alloc;
}

void reflect(InlineVariantType type, llvm::BumpPtrAllocator& alloc,
             std::vector<Type>& pool,
             std::unordered_map<const void*, int32_t>& cache) {
  auto* terms = reflectTerms(type->terms, alloc, pool, cache);
  Type typ{.tag = Type::Kind::InlineVariant};
  typ.value.InlineVariant = {
      .name = reflectName(type.name(), alloc),
      .terms = {terms, type->terms.size()},
      .term_offset = type->term_offset,
      .tag_offset = type->tag_offset,
      .tag_width = type->tag_width,
  };
  pool.emplace_back(typ);
}

#undef FOR_EACH_REFLECTABLE_PROTOJIT_TYPE

ValueType reflectableTypeFor(ValueType type) {
  if (type.isa<IntType>()) {
    return type;
  }
  if (auto array = type.dyn_cast<ArrayType>()) {
    types::Array ary{array};
    ary.elem = reflectableTypeFor(ary.elem);
    return ArrayType::get(array.getContext(), ary);
  }
  if (auto vec = type.dyn_cast<VectorType>()) {
    auto elem = reflectableTypeFor(vec->elem);
    return VectorType::get(
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
  if (auto str = type.dyn_cast<StructType>()) {
    llvm::SmallVector<types::StructField, 4> fields;
    for (auto& f : str->fields) {
      fields.emplace_back(types::StructField{
          .type = reflectableTypeFor(f.type),
          .name = f.name,
          .offset = f.offset,
      });
    }
    types::Struct result = str;
    result.fields = fields;
    auto typ = types::StructType::get(type.getContext(), TypeDomain::kReflect,
                                      str.name());
    typ.setTypeData(result);
    return typ;
  }
  if (auto var = type.dyn_cast<VariantType>()) {
    llvm::SmallVector<types::Term, 4> terms;
    Width term_max_size = Bits(0);
    for (auto& t : var.terms()) {
      auto back = terms.emplace_back(types::Term{
          .name = t.name,
          .type = reflectableTypeFor(t.type),
          .tag = t.tag,
      });
      term_max_size = std::max(term_max_size, back.type.headSize());
    }
    if (auto inl = type.dyn_cast<InlineVariantType>()) {
      types::InlineVariant result = inl;
      result.terms = terms;
      auto typ = types::InlineVariantType::get(
          type.getContext(), TypeDomain::kReflect, inl.name());
      typ.setTypeData(result);
      return typ;
    }
    if (auto outl = type.dyn_cast<OutlineVariantType>()) {
      types::InlineVariant result{
          .terms = terms,
          .term_offset = Bytes(0),
          .term_size = term_max_size,
          .tag_offset = term_max_size,
          .tag_width = outl.tag_width(),
          .size = term_max_size + outl.tag_width(),
          .alignment = outl->term_alignment,
      };
      auto typ = types::InlineVariantType::get(
          type.getContext(), TypeDomain::kReflect, outl.name());
      typ.setTypeData(result);
      return typ;
    }
  }
  UNREACHABLE();
}

}  // namespace reflect
}  // namespace pj
