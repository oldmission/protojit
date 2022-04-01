#pragma once

#include <malloc.h>

#include <vector>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include "types.hpp"
#include "util.hpp"

namespace pj {

struct Scoped {
  virtual ~Scoped();
};

class Scope {
 public:
  Scope(mlir::MLIRContext* context) : context_(context) {}
  ~Scope() {
    for (auto* x : allocations) {
      free(x);
    }
  }

  // TODO(2): proper incremental allocator
  Scoped* Allocate(size_t size) {
    auto* result = reinterpret_cast<Scoped*>(malloc(size));
    allocations.push_back(result);
    return result;
  }

  mlir::MLIRContext* Context() { return context_; }
  types::StructType Unit();

 private:
  std::vector<Scoped*> allocations;

  mlir::MLIRContext* context_;
  types::StructType unit_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(Scope);
};

}  // namespace pj

void* operator new(size_t size, pj::Scope& scope);
void* operator new(size_t size, pj::Scope* scope);
void* operator new[](size_t size, pj::Scope& scope);
void* operator new[](size_t size, pj::Scope* scope);
