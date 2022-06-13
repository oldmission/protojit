#include <iostream>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Operation.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include "pj/reflect.pj.hpp"
#include "types.hpp"

#ifndef NDEBUG
__attribute__((visibility("default"))) void dump(mlir::Operation* op) {
  op->dump();
}
__attribute__((visibility("default"))) void dumpResultType(
    mlir::Operation* op) {
  op->getResult(0).getType().dump();
}
__attribute__((visibility("default"))) void dump(llvm::Type* op) { op->dump(); }
__attribute__((visibility("default"))) void dump(llvm::Value* val) {
  val->dump();
}
__attribute__((visibility("default"))) void dump(llvm::BasicBlock* bb) {
  bb->dump();
}
__attribute__((visibility("default"))) void dumpType(llvm::Value* val) {
  val->getType()->dump();
}
__attribute__((visibility("default"))) void dump(llvm::Instruction* it) {
  it->dump();
}
__attribute__((visibility("default"))) void dumpMapping(
    mlir::DenseMap<void*, void*>* mapping) {
  for (auto& pair : *mapping) {
    std::cerr << pair.first << " -> " << pair.second << "\n";
  }
}
__attribute__((visibility("default"))) void printTree(
    pj::types::ValueType type) {
  type.printTree(llvm::errs());
}
__attribute__((visibility("default"))) auto getType(size_t ptr) {
  return *reinterpret_cast<pj::reflect::Type*>(ptr);
}
#endif
