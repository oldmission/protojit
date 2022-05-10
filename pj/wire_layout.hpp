#pragma once

#include "plan.hpp"

namespace pj {

class WireLayout : public TypePass {
 public:
  WireLayout(mlir::MLIRContext& ctx) : ctx_(ctx) {}

  bool run(types::Protocol& proto) override;

 private:
#define DEFINE_VISIT_FUNCTION(TYPE) types::ValueType visit(TYPE type);
  FOR_EACH_VALUE_TYPE(DEFINE_VISIT_FUNCTION)
#undef DEFINE_VISIT_FUNCTION

  template <typename Variant>
  types::ValueType visitVariant(Variant var);

  types::ValueType visit(mlir::Type type);

  mlir::MLIRContext& ctx_;
};

}  // namespace pj
