#include <unordered_set>

#include <llvm/Support/Debug.h>

#include <pj/util.hpp>

#include "convert_internal.hpp"
#include "defer.hpp"
#include "plan.hpp"
#include "array_ref.hpp"
#include "variant_outlining.hpp"
#include "vector_hoisting.hpp"
#include "wire_layout.hpp"

namespace pj {
using namespace types;

void TypePass::replaceStructField(StructType type, intptr_t index,
                                  types::ValueType field_type,
                                  llvm::StringRef field_name) const {
  auto data = Struct(type);

  ArrayRefConverter<StructField> field_conv{data.fields, data.fields.size()};
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

  ArrayRefConverter<Term> term_conv{data.terms, data.terms.size()};
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

#define DEBUG_TYPE "pj.plan"

ProtocolType plan_protocol(mlir::MLIRContext& ctx, mlir::Type type,
                           PathAttr path) {
  auto proto =
      Protocol{.head = type.cast<ValueType>(), .buffer_offset = Bytes(0)};

  ConvertInternal conv_internal{ctx};
  conv_internal.run(proto);

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "Before planning:\n"
                      "==================================================\n";
      proto.head.printTree(llvm::errs()));

  VariantOutlining outlining{ctx, path};
  outlining.run(proto);

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After variant outlining:\n"
                      "==================================================\n";
      proto.head.printTree(llvm::errs()));

  VectorHoisting hoisting{ctx};
  hoisting.run(proto);

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After vector hoisting:\n"
                      "==================================================\n";
      proto.head.printTree(llvm::errs()));

  WireLayout layout{ctx};
  layout.run(proto);

  OutlineVariantOffsetGeneration offset;
  offset.run(proto);

  LLVM_DEBUG(
      llvm::errs() << "==================================================\n"
                      "After wire layout and outline variant offset gen:\n"
                      "==================================================\n";
      proto.head.printTree(llvm::errs()));

  return ProtocolType::get(&ctx, proto);
}

}  // namespace pj
