#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <optional>
#include <string>
#include <vector>

#include "span.hpp"
#include "types.hpp"

namespace pj {

class TypePass {
 public:
  virtual bool run(types::Protocol& proto) = 0;

 protected:
  void replaceStructField(types::StructType type, intptr_t index,
                          types::ValueType field_type,
                          llvm::StringRef field_name = {}) const;

  void replaceVariantTerm(types::VariantType type, intptr_t index,
                          types::ValueType term_type,
                          llvm::StringRef term_name = {}) const;
};

types::ProtocolType plan_protocol(mlir::MLIRContext& ctx, mlir::Type type,
                                  types::PathAttr path);

}  // namespace pj
