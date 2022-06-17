#pragma once

#include <llvm/ADT/ilist.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

#include <functional>
#include <iterator>

#include "arch.hpp"
#include "span.hpp"
#include "type_domain.hpp"
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
namespace self {
struct Type;
}  // namespace self

namespace types {

struct ValueType;

using ChildVector = llvm::SmallVector<std::pair<mlir::Type, std::string>>;

struct ValueTypeStorage : public mlir::TypeStorage {
  virtual ~ValueTypeStorage() {}
  virtual void print(llvm::raw_ostream& os) const = 0;
  virtual bool hasDetails() const = 0;
  virtual void printDetails(llvm::raw_ostream& os) const = 0;
  virtual Width headSize() const = 0;
  virtual Width headAlignment() const = 0;
  virtual bool hasMaxSize() const = 0;
  virtual ChildVector children() const = 0;
  virtual bool isBinaryCompatibleWith(ValueType type) const = 0;
};

// Base class for all PJ types that represent values; i.e., everything except
// buffer and userstate.
struct ValueType : public mlir::Type {
  using Type::Type;

  static bool classof(mlir::Type val);

  void print(llvm::raw_ostream& os) const {
    static_cast<const ValueTypeStorage*>(impl)->print(os);
  }

  bool hasDetails() const {
    return static_cast<const ValueTypeStorage*>(impl)->hasDetails();
  }

  void printDetails(llvm::raw_ostream& os) const {
    static_cast<const ValueTypeStorage*>(impl)->printDetails(os);
  }

  void printTree(llvm::raw_ostream& os, llvm::StringRef name = {},
                 llvm::StringRef indent = {}, bool is_last = true) const;

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

  // Indicates whether the size taken up by an instance of this type as well as
  // all of its subelements is guaranteed to be bounded. For example,
  // char8[8:][4:8] would return false, because the inner vector is potentially
  // unbounded in total size, but char8[8:256][4] would return true.
  bool hasMaxSize() const {
    return static_cast<const ValueTypeStorage*>(impl)->hasMaxSize();
  }

  ChildVector children() const {
    return static_cast<const ValueTypeStorage*>(impl)->children();
  }

  bool isBinaryCompatibleWith(ValueType type) const {
    return static_cast<const ValueTypeStorage*>(impl)->isBinaryCompatibleWith(
        type);
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
  return Name{allocator.copyInto(Name{pieces.data(), pieces.size()})};
}

template <typename Data, typename Type>
struct StructuralTypeStorage : public ValueTypeStorage {
  using KeyTy = Data;

  StructuralTypeStorage(const Data& key) : key(key) {}

  bool operator==(const Data& k) const { return key == k; }

  static llvm::hash_code hashKey(const Data& k) {
    using ::llvm::hash_value;
    return hash_value(k);
  }

  static StructuralTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                          const Data& key) {
    return new (allocator.allocate<StructuralTypeStorage>())
        StructuralTypeStorage(type_intern(allocator, key));
  }

  void print(llvm::raw_ostream& os) const override { os << key; }
  bool hasDetails() const override { return key.hasDetails(); }
  void printDetails(llvm::raw_ostream& os) const override {
    key.printDetails(os);
  }
  Width headSize() const override { return key.headSize(); }
  Width headAlignment() const override { return key.headAlignment(); }
  bool hasMaxSize() const override { return key.hasMaxSize(); }
  ChildVector children() const override { return key.children(); }
  bool isBinaryCompatibleWith(ValueType type) const override {
    return type.isa<Type>() &&
           key.isBinaryCompatibleWith(Data(type.cast<Type>()));
  }

  KeyTy key;
};

template <typename D, typename T>
struct StructuralTypeBase : public ValueType {
  using ValueType::ValueType;

  operator const D&() const { return storage()->key; }

  const D* operator->() const { return &storage()->key; }

 private:
  const StructuralTypeStorage<D, T>* storage() const {
    return static_cast<const StructuralTypeStorage<D, T>*>(impl);
  }
};

struct NominalTypeStorageBase : public ValueTypeStorage {
  virtual ~NominalTypeStorageBase() {}
  virtual Name name() const = 0;
  virtual DomainAttr domain() const = 0;
};

template <typename Data, typename Type>
struct NominalTypeStorage : public NominalTypeStorageBase {
  using KeyTy = std::pair<DomainAttr, Name>;

  NominalTypeStorage(DomainAttr domain, Name name)
      : domain_(domain), name_(name) {}

  bool operator==(const KeyTy& k) const {
    return domain_ == std::get<DomainAttr>(k) && name_ == std::get<Name>(k);
  }

  static llvm::hash_code hashKey(const KeyTy& k) {
    using ::llvm::hash_combine;
    using ::llvm::hash_value;
    return hash_combine(hash_value(std::get<DomainAttr>(k)),
                        hash_value(std::get<Name>(k)));
  }

  static NominalTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                       const KeyTy& key) {
    return new (allocator.allocate<NominalTypeStorage>())
        NominalTypeStorage(key.first, type_intern(allocator, key.second));
  }

  void print(llvm::raw_ostream& os) const override {
    domain_.print(os);
    for (auto p : name_) {
      os << "::" << p;
    }
  }

  bool hasDetails() const override { return type_data_.hasDetails(); }
  void printDetails(llvm::raw_ostream& os) const override {
    type_data_.printDetails(os);
  }

  Width headSize() const override { return type_data_.headSize(); }
  Width headAlignment() const override { return type_data_.headAlignment(); }
  bool hasMaxSize() const override { return type_data_.hasMaxSize(); }
  ChildVector children() const override { return type_data_.children(); }
  bool isBinaryCompatibleWith(ValueType type) const override {
    return type.isa<Type>() &&
           type_data_.isBinaryCompatibleWith(Data(type.cast<Type>()));
  }

  mlir::LogicalResult mutate(mlir::TypeStorageAllocator& allocator,
                             const Data& type_data) {
    type_data_ = type_intern(allocator, type_data);
    return mlir::success();
  }

  Name name() const override { return name_; }

  DomainAttr domain() const override { return domain_; }

  DomainAttr domain_;
  Name name_;
  Data type_data_;
};

// Base class for all PJ types that have a name (derived from NominalTypeBase).
struct NominalType : public ValueType {
  using ValueType::ValueType;

  static bool classof(mlir::Type val);

  Name name() const { return storage()->name(); };

  DomainAttr domain() const { return storage()->domain(); };

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

  const D& getTypeData() const { return storage()->type_data_; }

  void setTypeData(const D& type_data) {
    (void)static_cast<T*>(this)->mutate(type_data);
  }

 private:
  const NominalTypeStorage<D, T>* storage() const {
    return static_cast<const NominalTypeStorage<D, T>*>(this->impl);
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
    auto* list = reinterpret_cast<llvm::StringRef*>(allocator.allocate(
        sizeof(llvm::StringRef) * key.size(), alignof(llvm::StringRef)));

    for (uintptr_t i = 0; i < key.size(); ++i) {
      list[i] = allocator.copyInto(key[i]);
    }

    return new (allocator.allocate<PathAttrStorage>())
        PathAttrStorage(Span<llvm::StringRef>{list, key.size()});
  }

  KeyTy key;
};

struct PathAttr : public mlir::Attribute::AttrBase<PathAttr, mlir::Attribute,
                                                   PathAttrStorage> {
  using Base::Base;
  using Base::get;

  template <typename OS>
  void print(OS& os) const {
    bool first = true;
    for (auto& part : getValue()) {
      if (!first) {
        os << ".";
      }
      os << std::string_view(part.data(), part.size());
      first = false;
    }
  }

  static PathAttr none(mlir::MLIRContext* C);
  static PathAttr fromString(mlir::MLIRContext* C, llvm::StringRef src_path);
  std::string toString() const;
  Span<llvm::StringRef> getValue() const { return getImpl()->key; }
  size_t unique_code() const { return reinterpret_cast<size_t>(impl); }

  bool empty() const { return getValue().size() == 0; }

  bool startsWith(llvm::StringRef prefix) const {
    auto& path = getImpl()->key;
    if (path.size() == 0) return false;
    return path[0] == prefix;
  }

  PathAttr narrow() const { return get(getContext(), getImpl()->key.slice(1)); }

  PathAttr into(llvm::StringRef prefix) const {
    if (startsWith(prefix)) {
      return narrow();
    }
    return none(getContext());
  }

  PathAttr expand(llvm::StringRef prefix) const;
};

}  // namespace types
}  // namespace pj
