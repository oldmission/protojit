#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <optional>
#include <string>
#include <vector>

#include "types.hpp"

namespace pj {

types::ValueType plan(mlir::MLIRContext& ctx, mlir::Type type,
                      types::PathAttr path);

}  // namespace pj
