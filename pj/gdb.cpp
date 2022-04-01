#include <mlir/IR/Operation.h>

#ifndef NDEBUG
__attribute__((visibility("default"))) void dump(mlir::Operation* op) {
  op->dump();
}
#endif
