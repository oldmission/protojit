#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/InliningUtils.h>

#include "concrete_types.hpp"
#include "exceptions.hpp"
#include "types.hpp"

namespace pj {
namespace ir {
using namespace mlir;
struct PJType;
struct WidthAttr;
}  // namespace ir
}  // namespace pj

#define GET_OP_CLASSES
#include "pj/ops.hpp.inc"

#define GET_OP_CLASSES
#include "pj/ir.hpp.inc"

namespace pj {
namespace ir {

using namespace mlir;

struct ProtoJitInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
    return true;
  }

  bool isLegalToInline(Region*, Region*, bool,
                       BlockAndValueMapping&) const final {
    return true;
  }

  void handleTerminator(Operation* op,
                        ArrayRef<Value> valuesToRepl) const final {
    auto returnOp = cast<RetOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto& it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }

  Operation* materializeCallConversion(OpBuilder& builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    throw InternalError("Should not need to convert.");
  }
};

class ProtoJitDialect : public Dialect {
 public:
  explicit ProtoJitDialect(MLIRContext* ctx);
  ~ProtoJitDialect();

  static llvm::StringRef getDialectNamespace() { return "pj"; }

  void printType(Type type, DialectAsmPrinter& printer) const override;
  void printAttribute(Attribute type,
                      DialectAsmPrinter& printer) const override;
};

struct PJTypeStorage : public TypeStorage {
  struct Params {
    const CType* const type;
    bool operator==(const Params& x) const { return type == x.type; }
    llvm::hash_code hash() const { return llvm::hash_combine(type); }
  };
  using KeyTy = Params;

  PJTypeStorage(const Params& params) : base_params(params) {}

  bool operator==(const KeyTy& key) const { return base_params == key; }

  static llvm::hash_code hashKey(const KeyTy& key) { return key.hash(); }

  static PJTypeStorage* construct(TypeStorageAllocator& allocator,
                                  const KeyTy& key) {
    return new (allocator.allocate<PJTypeStorage>()) PJTypeStorage(key);
  }

  Params base_params;
};

struct WidthAttributeStorage : public AttributeStorage {
  using KeyTy = Width;

  WidthAttributeStorage(Width value) : value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy& key) const { return key == value; }

  static llvm::hash_code hashKey(const KeyTy& key) { return key.bits(); }

  /// Construct a new storage instance.
  static WidthAttributeStorage* construct(AttributeStorageAllocator& allocator,
                                          const KeyTy& key) {
    return new (allocator.allocate<WidthAttributeStorage>())
        WidthAttributeStorage(key);
  }

  Width value;
};

struct WidthAttr
    : public Attribute::AttrBase<WidthAttr, Attribute, WidthAttributeStorage> {
  using Base::Base;
  using Base::get;
  Width* operator->() const { return &getImpl()->value; }
  Width getValue() const { return getImpl()->value; }
};

struct PJType : public Type::TypeBase<PJType, Type, PJTypeStorage> {
 public:
  using Base::Base;
  const CType* operator->() const { return getImpl()->base_params.type; }
  Type toLLVM() const;
};

struct UserStateType : public Type::TypeBase<UserStateType, Type, TypeStorage> {
  using Base::Base;
  Type toLLVM() const;

  static UserStateType get(MLIRContext* ctx) { return Base::get(ctx); }
};

struct ArrayTypeStorage : public PJTypeStorage {
  struct Params {
    const intptr_t length;
    bool operator==(const Params& x) const { return length == x.length; }

    llvm::hash_code hash() const { return llvm::hash_combine(length); }
  };
  using KeyTy = std::pair<PJTypeStorage::Params, Params>;

  ArrayTypeStorage(const PJTypeStorage::Params& base, const Params& params)
      : PJTypeStorage(base), params(params) {}

  bool operator==(const KeyTy& key) const {
    return key == std::make_pair(base_params, params);
  }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine(key.first.hash(), key.second.hash());
  }

  static ArrayTypeStorage* construct(TypeStorageAllocator& allocator,
                                     const KeyTy& key) {
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(key.first, key.second);
  }

  Params params;
};

// TODO: put this somewhere more appropriate
TypeRange ReplaceTerminators(ConversionPatternRewriter& _, Block* final,
                             Region::iterator begin, Region::iterator end,
                             bool update_join_args = true);

inline mlir::Type GetIndexType(mlir::OpBuilder& _) {
  return mlir::IndexType::get(_.getContext());
}

inline mlir::Value GetIndexConstant(const mlir::Location& L, mlir::OpBuilder& _,
                                    intptr_t size) {
  auto index_type = mlir::IndexType::get(_.getContext());
  auto index_attr = _.getIntegerAttr(index_type, size);
  return _.create<ConstantOp>(L, index_type, index_attr);
}

inline mlir::Value GetIntegerConstant(const mlir::Location& L,
                                      mlir::OpBuilder& _, Width width,
                                      intptr_t val) {
  auto type = _.getIntegerType(width.bits());
  auto attr = _.getIntegerAttr(type, val);
  return _.create<ConstantOp>(L, type, attr);
}

void printAttrForFunctionName(llvm::raw_ostream& os, mlir::Attribute attr);

}  // namespace ir
}  // namespace pj
