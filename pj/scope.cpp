#include "scope.hpp"

#include "abstract_types.hpp"
#include "concrete_types.hpp"

namespace pj {
Scoped::~Scoped() {}

AType* Scope::AUnit() {
  if (!aunit_) {
    aunit_ = new (this) AStructType({});
  }
  return aunit_;
}

CType* Scope::CUnit() {
  if (!cunit_) {
    cunit_ = new (this) CStructType(AUnit(), Bytes(0), Bytes(0), {});
  }
  return cunit_;
}

}  // namespace pj

void* operator new(size_t size, pj::Scope& scope) {
  return scope.Allocate(size);
}

void* operator new(size_t size, pj::Scope* scope) {
  return scope->Allocate(size);
}
