#pragma once

#include "plan.hpp"

namespace pj {

// Converts all NominalTypes to InternalDomain so that types can be freely
// mutated during planning without affecting existing types.
class ConvertInternal : public TypePass {
 public:
  ConvertInternal(mlir::MLIRContext& ctx) : ctx_(ctx) {}

  bool run(types::Protocol& proto) override;

 private:
  types::ValueType convert(types::VariantType type) const;
  types::ValueType convert(types::ValueType type) const;

  mlir::MLIRContext& ctx_;
};

}  // namespace pj
