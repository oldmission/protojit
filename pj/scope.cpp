#include "scope.hpp"

#include "types.hpp"

namespace pj {
Scoped::~Scoped() {}
}  // namespace pj

void* operator new(size_t size, pj::Scope& scope) {
  return scope.Allocate(size);
}

void* operator new(size_t size, pj::Scope* scope) {
  return scope->Allocate(size);
}

void* operator new[](size_t size, pj::Scope& scope) {
  return scope.Allocate(size);
}

void* operator new[](size_t size, pj::Scope* scope) {
  return scope->Allocate(size);
}
