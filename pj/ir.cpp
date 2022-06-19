#include "ir.hpp"

#include <llvm/Support/ScopedPrinter.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/DialectImplementation.h>

#include "types.hpp"

namespace pj {
using namespace types;
using namespace mlir;

namespace ir {

ProtoJitDialect::~ProtoJitDialect() {}

void ProtoJitDialect::printType(Type type, DialectAsmPrinter& p) const {
  if (type.isa<UserStateType>()) {
    p << "userstate";
  } else if (type.isa<HandlersArrayType>()) {
    p << "handlers";
  } else if (type.isa<ValueType>()) {
    type.cast<ValueType>().print(p.getStream());
  } else if (type.isa<BoundedBufferType>()) {
    type.cast<BoundedBufferType>().print(p.getStream());
  } else if (type.isa<RawBufferType>()) {
    type.cast<RawBufferType>().print(p.getStream());
  } else if (type.isa<DummyBufferType>()) {
    type.cast<DummyBufferType>().print(p.getStream());
  } else {
    UNREACHABLE();
  }
}

void ProtoJitDialect::printAttribute(Attribute attr,
                                     DialectAsmPrinter& p) const {
  if (auto width = attr.dyn_cast<WidthAttr>()) {
    width.print(p.getStream());
  } else if (auto domain = attr.dyn_cast<DomainAttr>()) {
    domain.print(p.getStream());
  } else if (auto path = attr.dyn_cast<PathAttr>()) {
    path.print(p.getStream());
  } else if (auto handler = attr.dyn_cast<DispatchHandlerAttr>()) {
    handler.print(p.getStream());
  } else {
    UNREACHABLE();
  }
}

ProtoJitDialect::ProtoJitDialect(MLIRContext* ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<ProtoJitDialect>()) {
  addAttributes<WidthAttr, HostDomainAttr, WireDomainAttr, ReflectDomainAttr,
                InternalDomainAttr, PathAttr, DispatchHandlerAttr>();

  addTypes<HandlersArrayType, UserStateType, UnitType, IntType, StructType,
           InlineVariantType, OutlineVariantType, ArrayType, VectorType,
           AnyType, ProtocolType, BoundedBufferType, RawBufferType,
           DummyBufferType>();

  addOperations<
#define GET_OP_LIST
#include "pj/ir.cpp.inc"
      >();
}

void printAttrForFunctionName(llvm::raw_ostream& os, mlir::Attribute attr) {
  if (auto path = attr.dyn_cast<PathAttr>()) {
    path.print(os);
  } else if (attr.isa<TypeAttr>()) {
    attr.print(os);
  } else {
    os << attr.getAsOpaquePointer();
  }
}

}  // namespace ir
}  // namespace pj

#include "pj/enums.cpp.inc"

#define GET_OP_CLASSES
#include "pj/ir.cpp.inc"
