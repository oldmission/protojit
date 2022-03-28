#include <llvm/ADT/ilist.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

#include "arch.hpp"

namespace pj {
namespace types {

struct ValueTypeStorage : public mlir::TypeStorage {
  virtual ~ValueTypeStorage(){};
  virtual void print(llvm::raw_ostream& os) const = 0;
  virtual Width head_size() const = 0;
};

// Base class for all PJ types that represent values; i.e., everything except
// buffer and userstate.
struct ValueType : public mlir::Type {
  using Type::Type;

  static bool classof(mlir::Type val);

  void print(llvm::raw_ostream& os) const {
    static_cast<const ValueTypeStorage*>(impl)->print(os);
  }

  Width head_size() const {
    return static_cast<const ValueTypeStorage*>(impl)->head_size();
  }

  size_t unique_code() const { return reinterpret_cast<size_t>(impl); }
};

template <typename T>
T type_intern(mlir::TypeStorageAllocator& allocator, const T& W);

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
  Width head_size() const override { return key.head_size(); }

  KeyTy key;
};

template <typename D, typename T>
struct StructuralTypeBase : public ValueType {
  using ValueType::ValueType;

  llvm::StringRef name() const { return T::getImpl()->name; }

  const D* operator->() const {
    auto* storage = static_cast<const StructuralTypeStorage<D>*>(impl);
    return &storage->key;
  }
};

enum class TypeDomain { kHost, kWire };

template <typename T>
struct NominalTypeStorage : public ValueTypeStorage {
  using KeyTy = std::pair<TypeDomain, llvm::StringRef>;

  NominalTypeStorage(const KeyTy& key) : key(key) {}

  bool operator==(const KeyTy& k) const { return key == k; }

  static llvm::hash_code hashKey(const KeyTy& k) {
    using ::llvm::hash_value;
    return hash_value(k);
  }

  static NominalTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                       const KeyTy& key) {
    return new (allocator.allocate<NominalTypeStorage>())
        NominalTypeStorage({key.first, allocator.copyInto(key.second)});
  }

  void print(llvm::raw_ostream& os) const override {
    switch (key.first) {
      case TypeDomain::kHost:
        os << "host::";
        break;
      case TypeDomain::kWire:
        os << "wire::";
        break;
    }
    os << key.second;
  }

  Width head_size() const override { return type_data_.head_size(); }

  mlir::LogicalResult mutate(mlir::TypeStorageAllocator& allocator,
                             const T& type_data) {
    type_data_ = type_intern(allocator, type_data);
    return mlir::success();
  }

  KeyTy key;
  T type_data_;
};

template <typename Base, typename D, typename T>
struct NominalTypeBase : public Base {
  using Base::Base;

  llvm::StringRef name() const { return this->T::getImpl()->name; }
  const D* operator->() const {
    return &static_cast<const NominalTypeStorage<D>*>(this->impl)->type_data_;
  }

  void setTypeData(const D& type_data) {
    (void)static_cast<T*>(this)->mutate(type_data);
  }
};

struct PathAttrStorage : public mlir::AttributeStorage {
  using KeyTy = llvm::ArrayRef<llvm::StringRef>;

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
    auto list = llvm::makeMutableArrayRef<llvm::StringRef>(
        reinterpret_cast<llvm::StringRef*>(allocator.allocate(
            sizeof(llvm::StringRef) * key.size(), alignof(llvm::StringRef))),
        key.size());

    for (intptr_t i = 0; i < key.size(); ++i) {
      list[i] = allocator.copyInto(key[i]);
    }

    return new (allocator.allocate<PathAttrStorage>()) PathAttrStorage(list);
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
  llvm::ArrayRef<llvm::StringRef> getValue() const { return getImpl()->key; }
  size_t unique_code() const { return reinterpret_cast<size_t>(impl); }

  PathAttr into(llvm::StringRef prefix) const {
    auto& path = getImpl()->key;
    if (path.size() == 0) {
      return *this;
    }
    return path[0] == prefix ? get(getContext(), path.slice(1)) : *this;
  }
};

}  // namespace types
}  // namespace pj
