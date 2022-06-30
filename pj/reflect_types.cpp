#include "reflect_types.hpp"
#include "protojit.hpp"

namespace pj {
namespace gen {

const PJVectorType* BuildPJType<reflect::ReflectionTypeVector>::build(
    PJContext* ctx, const PJDomain* domain) {
  return PJCreateVectorType(
      ctx,
      /*elem=*/BuildPJType<reflect::Type>::build(ctx, domain),
      /*min_length=*/0,
      /*max_length=*/kNone,
      /*wire_min_length=*/0,
      /*ppl_count=*/0,
      /*length_offset=*/offsetof(reflect::ReflectionTypeVector, size_) * 8,
      /*length_size=*/sizeof(reflect::ReflectionTypeVector::size_) * 8,
      /*ref_offset=*/offsetof(reflect::ReflectionTypeVector, offset_) * 8,
      /*ref_size=*/sizeof(reflect::ReflectionTypeVector::offset_) * 8,
      /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
      /*inline_payload_offset=*/kNone,
      /*inline_payload_size=*/kNone,
      /*partial_payload_offset=*/kNone,
      /*partial_payload_size=*/kNone,
      /*size=*/sizeof(reflect::ReflectionTypeVector) * 8,
      /*alignment=*/alignof(reflect::ReflectionTypeVector) * 8,
      /*outlined_payload_alignment=*/alignof(reflect::Type) * 8);
}

}  // namespace gen
}  // namespace pj
