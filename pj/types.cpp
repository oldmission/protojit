#include "types.hpp"
#include "util.hpp"

namespace pj {
namespace types {

Struct type_intern(mlir::TypeStorageAllocator& allocator, const Struct& key) {
  Struct s;

  auto fields = reinterpret_cast<StructField*>(allocator.allocate(
      sizeof(StructField) * key.fields.size(), alignof(StructField)));
  s.size = key.size;
  s.alignment = key.alignment;

  for (uintptr_t i = 0; i < key.fields.size(); ++i) {
    s.has_max_size = s.has_max_size && key.fields[i].type.hasMaxSize();
    fields[i] = StructField{
        .type = key.fields[i].type,
        .name = allocator.copyInto(key.fields[i].name),
        .offset = key.fields[i].offset,
    };

    if (key.fields[i].type.isa<OutlineVariantType>()) {
      assert(!s.outline_variant);
      s.outline_variant = key.fields[i].type;
    } else if (auto child = key.fields[i].type.dyn_cast<StructType>()) {
      if (child->outline_variant) {
        assert(!s.outline_variant);
        s.outline_variant = child->outline_variant;
      }
    }
  }

  s.fields = {&fields[0], key.fields.size()};
  return s;
}

}  // namespace types
}  // namespace pj