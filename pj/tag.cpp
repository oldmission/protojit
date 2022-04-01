#include "tag.hpp"

#include "concrete_types.hpp"

namespace pj {

const CType* CStructType::Resolve(PathPiece tag) const {
  if (tag.begin == tag.end || !fields.count(*tag.begin)) {
    return nullptr;
  }
  return fields.at(*tag.begin).type->Resolve(Tail(tag));
}

const CType* CVariantType::Resolve(PathPiece tag) const {
  return IsDotTag(tag) ? this : nullptr;
}

}  // namespace pj
