#include <unordered_set>

#include "convert_internal.hpp"
#include "defer.hpp"
#include "plan.hpp"
#include "span.hpp"
#include "util.hpp"
#include "variant_outlining.hpp"
#include "wire_layout.hpp"

namespace pj {

using namespace types;

void TypePass::replaceStructField(StructType type, intptr_t index,
                                  types::ValueType field_type,
                                  llvm::StringRef field_name) const {
  auto data = Struct(type);

  SpanConverter<StructField> field_conv{data.fields, data.fields.size()};
  if (!field_name.empty()) {
    field_conv.storage()[index].name = field_name;
  }
  field_conv.storage()[index].type = field_type;
  data.fields = field_conv.get();

  type.setTypeData(data);
}

template <typename V>
void replaceVariantTermGeneric(V type, intptr_t index, ValueType term_type,
                               llvm::StringRef term_name) {
  auto data = type.getTypeData();

  SpanConverter<Term> term_conv{data.terms, data.terms.size()};
  if (!term_name.empty()) {
    term_conv.storage()[index].name = term_name;
  }
  term_conv.storage()[index].type = term_type;
  data.terms = term_conv.get();

  type.setTypeData(data);
}

void TypePass::replaceVariantTerm(VariantType type, intptr_t index,
                                  ValueType term_type,
                                  llvm::StringRef term_name) const {
  if (auto in = type.dyn_cast<InlineVariantType>()) {
    replaceVariantTermGeneric(in, index, term_type, term_name);
  } else if (auto out = type.dyn_cast<OutlineVariantType>()) {
    replaceVariantTermGeneric(out, index, term_type, term_name);
  } else {
    UNREACHABLE();
  }
}

ProtocolType plan_protocol(mlir::MLIRContext& ctx, mlir::Type type,
                           PathAttr path) {
  auto proto =
      Protocol{.head = type.cast<ValueType>(), .buffer_offset = Bytes(0)};

  VariantOutlining outlining{ctx, path};
  outlining.run(proto);

  WireLayout layout{ctx};
  layout.run(proto);

  OutlineVariantOffsetGeneration offset;
  offset.run(proto);

  return ProtocolType::get(&ctx, proto);
}

}  // namespace pj
