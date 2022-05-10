#pragma once

#include <optional>

#include "plan.hpp"

namespace pj {

// Splits vectors with inline data into two - a fully inline vector and a fully
// outline vector - and merges the tag distinguishing the two with the parent
// outline variant.
//
// A simple example of this optimization is shown as follows.
// Original:
//
//   struct A { vec: char8[8:256]; }
//   variant B { a: A; ... }
//   protocol BProto : B @_;
//
// Optimization:
//
//   struct AShort { vec: char8[8:8]; }
//   struct ALong { vec: char8[0:256]; }
//   variant B { a: AShort; a: ALong; ... }
//   protocol BProto : B @_;
//
// This makes the common case of AShort significantly faster because the entire
// inline data can be unconditionally copied with no extra cost instead of
// checking for the length and looping.
//
// This optimization must be followed by Packing, because it emits types whose
// layouts are not fully specified.
class VectorHoisting : public TypePass {
 public:
  VectorHoisting(mlir::MLIRContext& ctx) : ctx_(ctx) {}

  bool run(types::Protocol& proto) override;

 private:
  struct Split {
    intptr_t inline_length;
    types::ValueType short_type;
    types::ValueType long_type;
    types::PathAttr path;
  };

  std::optional<Split> splitFirstEligibleVector(mlir::Type type);
  void hoistVectors(types::OutlineVariantType var);
  bool findOutlineAndHoist(mlir::Type type);

  mlir::MLIRContext& ctx_;
};

}  // namespace pj
