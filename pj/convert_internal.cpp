#include "convert_internal.hpp"

namespace pj {
using namespace types;

ValueType ConvertInternal::convert(ValueType type) const {
  if (auto str = type.dyn_cast<StructType>()) {
    auto new_str =
        StructType::get(&ctx_, InternalDomainAttr::get(&ctx_), str.name());
    new_str.setTypeData(str.getTypeData());
    for (uintptr_t i = 0; i < str->fields.size(); ++i) {
      replaceStructField(new_str, i, convert(str->fields[i].type));
    }
    return new_str;
  }

  if (auto var = type.dyn_cast<InlineVariantType>()) {
    auto new_var = InlineVariantType::get(&ctx_, InternalDomainAttr::get(&ctx_),
                                          var.name());
    new_var.setTypeData(var.getTypeData());
    for (uintptr_t i = 0; i < var->terms.size(); ++i) {
      replaceVariantTerm(new_var, i, convert(var->terms[i].type));
    }
    return new_var;
  }

  if (auto ary = type.dyn_cast<ArrayType>()) {
    auto data = Array(ary);
    data.elem = convert(ary->elem);
    return ArrayType::get(&ctx_, data);
  }

  if (auto vec = type.dyn_cast<VectorType>()) {
    auto data = Vector(vec);
    data.elem = convert(vec->elem);
    return VectorType::get(&ctx_, data);
  }

  assert(!type.isa<NominalType>());
  return type;
}

bool ConvertInternal::run(Protocol& proto) {
  proto.head = convert(proto.head);
  return true;
}

}  // namespace pj
