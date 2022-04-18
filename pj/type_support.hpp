#pragma once

#include <llvm/ADT/ilist.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

#include <functional>

#include "arch.hpp"
#include "span.hpp"
#include "util.hpp"

namespace pj {
namespace types {

// Contains the pieces of a fully qualified name
using Name = Span<llvm::StringRef>;

}  // namespace types
}  // namespace pj

namespace llvm {

inline bool operator<(const pj::types::Name a, const pj::types::Name b) {
  uintptr_t limit = std::min(a.size(), b.size());
  for (uintptr_t i = 0; i < limit; ++i) {
    int cmp = a[i].compare(b[i]);
    if (cmp != 0) {
      return cmp == -1;
    }
  }
  return a.size() < b.size();
}

}  // namespace llvm

namespace pj {
namespace types {

struct ValueTypeStorage : public mlir::TypeStorage {
  virtual ~ValueTypeStorage(){};
  virtual void print(llvm::raw_ostream& os) const = 0;
  virtual Width headSize() const = 0;
  virtual Width headAlignment() const = 0;
};

// Base class for all PJ types that represent values; i.e., everything except
// buffer and userstate.
struct ValueType : public mlir::Type {
  using Type::Type;

  static bool classof(mlir::Type val);

  void print(llvm::raw_ostream& os) const {
    static_cast<const ValueTypeStorage*>(impl)->print(os);
  }

  bool isUnit() const {
    // The pjc parser represents unit types via nullptr.
    return impl == nullptr || headSize() == Bits(0);
  }

  bool isEnum() const;

  Width headSize() const {
    return static_cast<const ValueTypeStorage*>(impl)->headSize();
  }

  Width headAlignment() const {
    return static_cast<const ValueTypeStorage*>(impl)->headAlignment();
  }

  size_t unique_code() const { return reinterpret_cast<size_t>(impl); }
};

template <typename T>
T type_intern(mlir::TypeStorageAllocator& allocator, const T& W);

inline Name type_intern(mlir::TypeStorageAllocator& allocator, Name n) {
  std::vector<llvm::StringRef> pieces;
  for (llvm::StringRef piece : n) {
    pieces.push_back(allocator.copyInto(piece));
  }
  return Name{allocator.copyInto(Name{&pieces[0], pieces.size()})};
}

template <typename T>
struct StructuralTypeStorage : public ValueTypeStorage {
  using KeyTy = T;

  StructuralTypeStorage(const T& key) : key(key) {}

  bool operator==(const T& k) const { return key == k; }

  static llvm::hash_code hashKey(const T& k) {
    using ::llvm::hash_value;
    return hash_value(k);
  }

  static StructuralTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                          const T& key) {
    return new (allocator.allocate<StructuralTypeStorage>())
        StructuralTypeStorage(type_intern(allocator, key));
  }

  void print(llvm::raw_ostream& os) const override { os << key; }
  Width headSize() const override { return key.headSize(); }
  Width headAlignment() const override { return key.headAlignment(); }

  KeyTy key;
};

template <typename D, typename T>
struct StructuralTypeBase : public ValueType {
  using ValueType::ValueType;

  operator const D&() const { return storage()->key; }

  const D* operator->() const { return &storage()->key; }

 private:
  const StructuralTypeStorage<D>* storage() const {
    return static_cast<const StructuralTypeStorage<D>*>(impl);
  }
};

enum class TypeDomain { kHost, kWire };

struct NominalTypeStorageBase : public ValueTypeStorage {
  virtual ~NominalTypeStorageBase() {}
  virtual Name name() const = 0;
  virtual TypeDomain type_domain() const = 0;
};

template <typename T>
struct NominalTypeStorage : public NominalTypeStorageBase {
  using KeyTy = std::pair<TypeDomain, Name>;

  NominalTypeStorage(TypeDomain type_domain, Name name)
      : type_domain_(type_domain), name_(name) {}

  bool operator==(const KeyTy& k) const {
    return type_domain_ == std::get<TypeDomain>(k) &&
           name_ == std::get<Name>(k);
  }

  static llvm::hash_code hashKey(const KeyTy& k) {
    using ::llvm::hash_combine;
    using ::llvm::hash_value;
    return hash_combine(hash_value(std::get<TypeDomain>(k)),
                        hash_value(std::get<Name>(k)));
  }

  static NominalTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                       const KeyTy& key) {
    return new (allocator.allocate<NominalTypeStorage>())
        NominalTypeStorage(key.first, type_intern(allocator, key.second));
  }

  void print(llvm::raw_ostream& os) const override {
    switch (type_domain_) {
      case TypeDomain::kHost:
        os << "host";
        break;
      case TypeDomain::kWire:
        os << "wire";
        break;
    }

    for (auto p : name_) {
      os << "::" << p;
    }
  }

  Width headSize() const override { return type_data_.headSize(); }
  Width headAlignment() const override { return type_data_.headAlignment(); }

  mlir::LogicalResult mutate(mlir::TypeStorageAllocator& allocator,
                             const T& type_data) {
    type_data_ = type_intern(allocator, type_data);
    return mlir::success();
  }

  Name name() const override { return name_; }

  TypeDomain type_domain() const override { return type_domain_; }

  TypeDomain type_domain_;
  Name name_;
  T type_data_;
};

// Base class for all PJ types that have a name (derived from NominalTypeBase).
struct NominalType : public ValueType {
  using ValueType::ValueType;

  static bool classof(mlir::Type val);

  Name name() const { return storage()->name(); };

  TypeDomain type_domain() const { return storage()->type_domain(); };

 private:
  const NominalTypeStorageBase* storage() const {
    return static_cast<const NominalTypeStorageBase*>(impl);
  }
};

template <typename Base, typename D, typename T>
struct NominalTypeBase : public Base {
  static_assert(std::is_base_of_v<NominalType, Base>);
  using Base::Base;

  operator const D&() const { return storage()->type_data_; }

  const D* operator->() const { return &storage()->type_data_; }

  void setTypeData(const D& type_data) {
    (void)static_cast<T*>(this)->mutate(type_data);
  }

 private:
  const NominalTypeStorage<D>* storage() const {
    return static_cast<const NominalTypeStorage<D>*>(this->impl);
  }
};

struct PathAttrStorage : public mlir::AttributeStorage {
  using KeyTy = Span<llvm::StringRef>;

  PathAttrStorage(KeyTy key) : key(key) {}

  bool operator==(const KeyTy& k) const { return key == k; }

  static llvm::hash_code hashKey(const KeyTy& k) {
    using ::llvm::hash_value;
    return hash_value(k);
  }

  static PathAttrStorage* construct(
      mlir::AttributeStorageAllocator& allocator) {
    return construct(allocator, KeyTy{});
  }

  // TODO: when constructing a PathAttr representing a tail of another
  // PathAttr, don't re-copy all the strings into the allocator.
  static PathAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                    KeyTy key) {
    auto list = reinterpret_cast<llvm::StringRef*>(allocator.allocate(
        sizeof(llvm::StringRef) * key.size(), alignof(llvm::StringRef)));

    for (uintptr_t i = 0; i < key.size(); ++i) {
      list[i] = allocator.copyInto(key[i]);
    }

    return new (allocator.allocate<PathAttrStorage>())
        PathAttrStorage(Span<llvm::StringRef>{&list[0], key.size()});
  }

  KeyTy key;
};

struct PathAttr : public mlir::Attribute::AttrBase<PathAttr, mlir::Attribute,
                                                   PathAttrStorage> {
  using Base::Base;
  using Base::get;

  void print(llvm::raw_ostream& os) const;
  static PathAttr none(mlir::MLIRContext* C);
  static PathAttr fromString(mlir::MLIRContext* C, llvm::StringRef src_path);
  std::string toString() const;
  Span<llvm::StringRef> getValue() const { return getImpl()->key; }
  size_t unique_code() const { return reinterpret_cast<size_t>(impl); }

  bool startsWith(llvm::StringRef prefix) const {
    auto& path = getImpl()->key;
    if (path.size() == 0) return false;
    return path[0] == prefix;
  }

  PathAttr into(llvm::StringRef prefix) const {
    if (startsWith(prefix)) {
      return get(getContext(), getImpl()->key.slice(1));
    }
    return none(getContext());
  }
};

}  // namespace types
}  // namespace pj
