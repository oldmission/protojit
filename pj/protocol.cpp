#include "protocol.hpp"

namespace pj {

Width Protocol::SizeOf(const CType* from, const Path& path) const {
  const auto head_size = head->ImpliedSize(from, path, tag);

  if (head_size.IsNone()) {
    return Width::None();
  }

  auto tag_type = head->Resolve(tag);
  const Width tag_size =
      tag_type != nullptr ? tag_type->AsVariant()->tag_size : Bits(0);

  return head_size + tag_size;
}

}  // namespace pj
