#pragma once

#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/Attributes.h>

#include <cstddef>
#include <variant>

// Type domains, indicating what domain the type is meant to be used.
// - Host: local in-memory types
// - Wire: serialized wire types
// - Reflect: types of values decoded into Any types
// - Internal: intermediate types generated during planning

#define FOR_EACH_DOMAIN(V) \
  V(Host)                  \
  V(Wire)                  \
  V(Reflect)               \
  V(Internal)

namespace pj {
namespace types {

struct HostDomain {
  bool operator==(const HostDomain&) const { return true; }
  void print(llvm::raw_ostream& os) const { os << "host"; }
};

struct WireDomain {
  bool operator==(const WireDomain& o) const { return id == o.id; }
  void print(llvm::raw_ostream& os) const { os << "wire(" << id << ")"; }

  static WireDomain get() { return WireDomain(counter++); }
  const size_t id;

 private:
  WireDomain(size_t id) : id(id) {}
  static size_t counter;
};

struct ReflectDomain {
  bool operator==(const ReflectDomain& o) const { return id == o.id; }
  void print(llvm::raw_ostream& os) const { os << "reflect(" << id << ")"; }

  static ReflectDomain get() { return ReflectDomain(counter++); }
  const size_t id;

 private:
  ReflectDomain(size_t id) : id(id) {}
  static size_t counter;
};

struct InternalDomain {
  bool operator==(const InternalDomain&) const { return true; }
  void print(llvm::raw_ostream& os) const { os << "internal"; }
};

struct DomainAttrStorageBase : public mlir::AttributeStorage {
  virtual ~DomainAttrStorageBase() {}
  virtual void print(llvm::raw_ostream& os) const = 0;
};

template <typename Domain>
struct DomainAttrStorage : public DomainAttrStorageBase {
  using KeyTy = Domain;

  DomainAttrStorage(const Domain& d) : domain(d) {}

  bool operator==(const Domain& d) const { return domain == d; }

  static llvm::hash_code hashKey(const Domain& domain) {
    if constexpr (std::is_same_v<Domain, HostDomain> ||
                  std::is_same_v<Domain, InternalDomain>) {
      return 0;
    } else if constexpr (std::is_same_v<Domain, WireDomain> ||
                         std::is_same_v<Domain, ReflectDomain>) {
      return ::llvm::hash_value(domain.id);
    }
  }

  static DomainAttrStorage* construct(
      mlir::AttributeStorageAllocator& allocator, Domain domain) {
    return new (allocator.allocate<DomainAttrStorage>())
        DomainAttrStorage(domain);
  }

  void print(llvm::raw_ostream& os) const { domain.print(os); }

  Domain domain;
};

struct DomainAttr : public mlir::Attribute {
  using Attribute::Attribute;

  void print(llvm::raw_ostream& os) const {
    static_cast<const DomainAttrStorageBase*>(impl)->print(os);
  }
  static bool classof(mlir::Attribute attr);
};

struct HostDomainAttr
    : public mlir::Attribute::AttrBase<HostDomainAttr, DomainAttr,
                                       DomainAttrStorage<HostDomain>> {
  using Base::Base;
  using Base::get;
};

struct WireDomainAttr
    : public mlir::Attribute::AttrBase<WireDomainAttr, DomainAttr,
                                       DomainAttrStorage<WireDomain>> {
  using Base::Base;
  using Base::get;

  static WireDomainAttr unique(mlir::MLIRContext* ctx) {
    return get(ctx, WireDomain::get());
  }
  size_t id() { return getImpl()->domain.id; }
};

struct ReflectDomainAttr
    : public mlir::Attribute::AttrBase<ReflectDomainAttr, DomainAttr,
                                       DomainAttrStorage<ReflectDomain>> {
  using Base::Base;
  using Base::get;

  static ReflectDomainAttr unique(mlir::MLIRContext* ctx) {
    return get(ctx, ReflectDomain::get());
  }
  size_t id() { return getImpl()->domain.id; }
};

struct InternalDomainAttr
    : public mlir::Attribute::AttrBase<InternalDomainAttr, DomainAttr,
                                       DomainAttrStorage<InternalDomain>> {
  using Base::Base;
  using Base::get;
};

}  // namespace types
}  // namespace pj
