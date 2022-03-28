#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace pj {

std::unique_ptr<mlir::Pass> createGeneratePass();

}  // namespace pj
