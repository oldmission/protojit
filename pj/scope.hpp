#pragma once

#include <malloc.h>

#include <vector>

#include "util.hpp"

namespace pj {

class AType;
class CType;

struct Scoped {
  virtual ~Scoped();
};

class Scope {
 public:
  Scope() {}
  ~Scope() {
    for (auto* x : allocations) {
      delete x;
    }
  }

  // TODO(2): proper incremental allocator
  Scoped* Allocate(size_t size) {
    auto* result = reinterpret_cast<Scoped*>(malloc(size));
    allocations.push_back(result);
    return result;
  }

  AType* AUnit();
  CType* CUnit();

 private:
  std::vector<Scoped*> allocations;

  AType* aunit_ = nullptr;
  CType* cunit_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(Scope);
};

}  // namespace pj

void* operator new(size_t size, pj::Scope& scope);
void* operator new(size_t size, pj::Scope* scope);
