#include <iostream>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Operation.h>

#ifndef NDEBUG
__attribute__((visibility("default"))) void dump(mlir::Operation* op) {
  op->dump();
}
__attribute__((visibility("default"))) void dump(size_t op) {
  ((mlir::Operation*)(op))->dump();
}
__attribute__((visibility("default"))) void dumpMapping(
    mlir::DenseMap<void*, void*>* mapping) {
  for (auto& pair : *mapping) {
    std::cerr << pair.first << " -> " << pair.second << "\n";
  }
}
#endif
