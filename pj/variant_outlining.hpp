#pragma once

#include "plan.hpp"

namespace pj {

// Converts the inline variant pointed to by path to an outline variant. Must
// be followed by WireLayout to fill in the newly generated OutlineVariant's
// layout and by OutlineVariantOffsetGeneration, when all other layout changes
// are finished, to set the term_offset.
class VariantOutlining : public TypePass {
 public:
  VariantOutlining(mlir::MLIRContext& ctx, types::PathAttr path)
      : ctx_(ctx), path_(path) {}

  bool run(types::Protocol& proto) override;

 private:
  types::ValueType tryOutlineVariant(mlir::Type type, types::PathAttr path);

  mlir::MLIRContext& ctx_;
  types::PathAttr path_;
};

// Generates the value of term_offset for the OutlineVariant and ensures that
// it is aligned to term_alignment. Must come after all layout changes because
// any further layout change can invalidate term_alignment.
class OutlineVariantOffsetGeneration : public TypePass {
 public:
  OutlineVariantOffsetGeneration() : outline_{} {}

  bool run(types::Protocol& proto) override;

 private:
  void incrementTermOffset(Width val);
  void run(mlir::Type type);

  types::OutlineVariantType outline_;
};

}  // namespace pj
