#pragma once

#include <optional>

#include "plan.hpp"

namespace pj {

// Simple test optimization that adds an additional 8-bit value option in a
// variant for when the value is small, to save space. Only modifies one int in
// the entire protocol, because modifying more of them is too much work for a
// throwaway test optimization.
//
// A simple example of this optimization is shown as follows.
// Original:
//
//   struct A { num: uint64; }
//   variant B { a: A; ... }
//
// Optimization:
//
//   struct AShort { num: uint8; }
//   struct AOriginal { num: uint64; }
//   variant B { a: AShort; a: AOriginal; ... }
//
// This optimization must be followed by Packing, because it emits types whose
// layouts are not fully specified.
class IntShortening : public TypePass {
 public:
  IntShortening(mlir::MLIRContext& ctx) : ctx_(ctx) {}

  bool run(types::Protocol& proto) override;

 private:
  struct Split {
    uint64_t threshold;
    types::ValueType short_type;
    types::ValueType original_type;
    types::PathAttr path;
  };

  std::optional<Split> splitFirstEligibleInt(mlir::Type type);
  bool shortenInt(types::VariantType var);
  bool findVariantAndShorten(mlir::Type type);

  mlir::MLIRContext& ctx_;
};

}  // namespace pj
