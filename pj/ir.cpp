#include "ir.hpp"

#include <llvm/Support/ScopedPrinter.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/DialectImplementation.h>

#include "concrete_types.hpp"
#include "types.hpp"

namespace pj {
using namespace types;
using namespace mlir;

Type CType::toIR(MLIRContext* C) const {
  return pj::ir::PJType::get(C, ir::PJTypeStorage::Params{.type = this});
}

namespace ir {

ProtoJitDialect::~ProtoJitDialect() {}

void ProtoJitDialect::printType(Type type, DialectAsmPrinter& P) const {
  if (type.isa<PJType>()) {
    auto pjtype = type.cast<PJType>();
    P << "val:" << pjtype->total_size().bits() << "/"
      << pjtype->alignment().bits();
  } else if (type.isa<UserStateType>()) {
    P << "userstate";
  } else if (type.isa<ValueType>()) {
    type.cast<ValueType>().print(P.getStream());
  } else if (type.isa<types::RawBufferType>()) {
    type.cast<types::RawBufferType>().print(P.getStream());
  } else if (type.isa<types::BoundedBufferType>()) {
    type.cast<types::BoundedBufferType>().print(P.getStream());
  } else {
    UNREACHABLE();
  }
}

void ProtoJitDialect::printAttribute(Attribute attr,
                                     DialectAsmPrinter& P) const {
  if (auto width = attr.dyn_cast<WidthAttr>()) {
    P << "b" << width->bits();
  } else if (auto path = attr.dyn_cast<PathAttr>()) {
    path.print(P.getStream());
  } else if (auto handler = attr.dyn_cast<DispatchHandlerAttr>()) {
    auto& os = P.getStream();
    handler.path().print(os);
    os << " -> " << handler.address();
  } else {
    UNREACHABLE();
  }
}

ProtoJitDialect::ProtoJitDialect(MLIRContext* ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<ProtoJitDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "pj/ops.cpp.inc"
      >();

  addAttributes<WidthAttr, PathAttr, DispatchHandlerAttr>();
  addInterfaces<ProtoJitInlinerInterface>();

  addTypes<PJType, UserStateType>();

  addOperations<
#define GET_OP_LIST
#include "pj/ir.cpp.inc"
      >();

  addTypes<types::IntType, types::StructType, types::InlineVariantType,
           types::OutlineVariantType, types::ArrayType, types::VectorType,
           types::AnyType, types::ProtocolType, types::RawBufferType,
           types::BoundedBufferType>();
}

static void print(OpAsmPrinter& printer, XIntOp op) {
  printer << "pj.xint(" << op.from() << ", " << op.to() << ")";
  printer.printOptionalAttrDict(op->getAttrs());
}

static void print(OpAsmPrinter& printer, LRefOp op) {
  printer << "pj.lref(" << op.base() << ", " << op.ref_offset().bits() << ", "
          << op.ref_size().bits() << ")";
  printer.printOptionalAttrDict(op->getAttrs());
}

static void print(OpAsmPrinter& p, XStrOp op) {
  p << "pj.xstr(" << op.from() << ", " << op.to() << ")";
  p.printOptionalAttrDict(op->getAttrs());

  Region& body = op.body();
  if (!body.empty()) {
    p.printRegion(body, /*printEntryBlockArgs=*/true);
  }
}

static void print(OpAsmPrinter& p, SStrOp op) {
  p << "pj.sstr(" << op.source() << ")";
  p.printOptionalAttrDict(op->getAttrs());

  Region& body = op.body();
  if (!body.empty()) {
    p.printRegion(body, /*printEntryBlockArgs=*/true);
  }
}

static void print(OpAsmPrinter& p, SListOp op) {
  p << "pj.slst(" << op.source() << ")";
  p.printOptionalAttrDict(op->getAttrs());

  Region& body = op.body();
  if (!body.empty()) {
    p.printRegion(body, /*printEntryBlockArgs=*/true);
  }
}

static void print(OpAsmPrinter& p, IterOp op) {
  p << "pj.iter[" << op.start() << " -> " << op.end() << "]";

  p << "(";
  for (uintptr_t i = 0; i < op.induction_variables().size(); ++i) {
    if (i != 0) p << ", ";
    p << op.induction_variables()[i];
  }
  p << ")";

  p.printOptionalAttrDict(op->getAttrs());

  Region& body = op.body();
  if (!body.empty()) {
    p.printRegion(body, /*printEntryBlockArgs=*/true);
  }
}

static void print(OpAsmPrinter& p, ProjectOp op) {
  p << "pj.proj(" << op.value() << ", " << op.byte_offset()
    << "): " << op.getType();
}

static void print(OpAsmPrinter& p, IIntOp op) {
  p << "pj.iint(" << op.to() << ")";
  p.printOptionalAttrDict(op->getAttrs());
}

static void print(OpAsmPrinter& p, MatchVariantOp op) {
  p << "pj.match(" << op.from() << ")";
  p.printOptionalAttrDict(op->getAttrs());
  Region& body = op.body();
  if (!body.empty()) {
    p.printRegion(body, /*printEntryBlockArgs=*/true);
  }
}

static void print(OpAsmPrinter& p, ETagOp op) {
  p << "pj.etag(" << op.to() << ", " << op.tag() << ")";
  p.printOptionalAttrDict(op->getAttrs());
}

static void print(OpAsmPrinter& p, DTagOp op) {
  p << "pj.dtag(" << op.from() << "):" << op.getType();
  p.printOptionalAttrDict(op->getAttrs());
}

static void print(OpAsmPrinter& p, DispatchOp op) {
  p << "pj.disp: " << llvm::HexNumber(op.target()) << "(" << op.value() << ", "
    << op.state() << ")";
}

static void print(OpAsmPrinter& p, LTagOp op) {
  p << "pj.ltag(" << op.from() << "[" << op.byte_offset()
    << "]) : " << op.getType();
  p.printOptionalAttrDict(op->getAttrs());
}

llvm::Optional<MutableOperandRange>  //
BTagOp::getMutableSuccessorOperands(unsigned index) {
  return llvm::None;
}

static void print(OpAsmPrinter& p, XArrayOp op) {
  p << "pj.xary(" << op.from() << ", " << op.to() << ")";
  p.printOptionalAttrDict(op->getAttrs());

  if (!op.xvalue().empty()) {
    p << " value:";
    p.printRegion(op.xvalue(), /*printEntryBlockArgs=*/true);
  }
  if (!op.xdefault().empty()) {
    p << " default:";
    p.printRegion(op.xdefault(), /*printEntryBlockArgs=*/true);
  }
}

static void print(OpAsmPrinter& p, SIntOp op) {
  p << "pj.sint(" << op.source() << ") : " << op.getType();
  p.printOptionalAttrDict(op->getAttrs());
}

static void print(OpAsmPrinter& p, CastOp op) {
  p << "pj.cast(" << op->getOperand(0) << ") : " << op.getType();
  p.printOptionalAttrDict(op->getAttrs());
}

TypeRange ReplaceTerminators(ConversionPatternRewriter& _, Block* final,
                             Region::iterator begin, Region::iterator end,
                             bool update_join_args) {
  TypeRange types;
  for (; begin != end; ++begin) {
    auto* term = &begin->back();
    if (isa<RetOp>(term)) {
      RetOp ret(term);

      _.setInsertionPointToEnd(&*begin);
      _.create<BranchOp>(ret.getLoc(), final, ret.getOperands());
      _.eraseOp(ret);

      types = ret.getOperandTypes();
    }
  }

  // Multiple outputs are not yet handled.
  if (update_join_args) {
    assert(final->getNumArguments() == 0);
    assert(types.size() < 2);
    if (types.size() > 0) {
      final->addArguments(types);
    }
  }

  return types;
}

OpFoldResult CastOp::fold(ArrayRef<Attribute> cstOperands) {
  auto cast = getOperand().getDefiningOp<CastOp>();
  if (cast && cast.getOperand().getType() == getType()) {
    return cast.getOperand();
  }

  return {};
}

}  // namespace ir

Value CListType::LoadLength(Location loc, Value value, OpBuilder& _) const {
  return _.create<ir::LTagOp>(loc, ir::GetIndexType(_), value,
                              ir::GetIndexConstant(loc, _, len_offset.bytes()),
                              len_size);
}

Value CListType::LoadOutlinedArray(Location loc, Value value, Type inner,
                                   OpBuilder& _) const {
  return _.create<ir::LRefOp>(loc, inner, value, ref_offset, ref_size);
}

void ir::printAttrForFunctionName(llvm::raw_ostream& os, mlir::Attribute attr) {
  if (attr.isa<PathAttr>()) {
    attr.print(os);
  }
}

}  // namespace pj

#define GET_OP_CLASSES
#include "pj/ops.cpp.inc"

#define GET_OP_CLASSES
#include "pj/ir.cpp.inc"
