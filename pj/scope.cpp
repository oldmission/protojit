#include "scope.hpp"

#include "types.hpp"

namespace pj {
Scoped::~Scoped() {}

types::StructType Scope::Unit() {
  if (!unit_) {
    unit_ =
        types::StructType::get(context_, types::TypeDomain::kHost, "<unit>");
    unit_.setTypeData(types::Struct{
        .fields = llvm::ArrayRef<types::StructField>{nullptr, 0ul},
        .size = Bytes(0),
        .alignment = Bytes(0)});
  }
  return unit_;
}

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
