#include "concrete_types.hpp"
#include "ir.hpp"

namespace pj {
using namespace ir;

mlir::Value CIntType::GenDefault(mlir::MLIRContext* C,
                                 const mlir::Location& loc, mlir::OpBuilder& _,
                                 const mlir::Value& to) const {
  _.create<IIntOp>(loc, to);
  return {};
}

mlir::Value CStructType::GenDefault(mlir::MLIRContext* C,
                                    const mlir::Location& loc,
                                    mlir::OpBuilder& _,
                                    const mlir::Value& to) const {
  // TODO: Test this.
  return {};
}

mlir::Value CVariantType::GenDefault(mlir::MLIRContext* C,
                                     const mlir::Location& loc,
                                     mlir::OpBuilder& _,
                                     const mlir::Value& to) const {
  auto tag_type = _.getIntegerType(tag_size.bits());
  auto tag_attr = _.getIntegerAttr(tag_type, kUndefTag);
  Value tag_val = _.create<ConstantOp>(loc, tag_type, tag_attr);
  if (this->tag_offset.IsNotNone()) {
    _.create<ETagOp>(loc, to, tag_val, this->tag_offset);
    return {};
  } else {
    return tag_val;
  }
}

mlir::Value CArrayType::GenDefault(mlir::MLIRContext* C,
                                   const mlir::Location& loc,
                                   mlir::OpBuilder& _,
                                   const mlir::Value& to) const {
  throw IssueError(19);
}

mlir::Value CListType::GenDefault(mlir::MLIRContext* C,
                                  const mlir::Location& loc, mlir::OpBuilder& _,
                                  const mlir::Value& to) const {
  throw IssueError(19);
}

mlir::Value CAnyType::GenDefault(mlir::MLIRContext* C,
                                 const mlir::Location& loc, mlir::OpBuilder& _,
                                 const mlir::Value& to) const {
  throw IssueError(19);
}

mlir::Value COutlinedType::GenDefault(mlir::MLIRContext* C,
                                      const mlir::Location& loc,
                                      mlir::OpBuilder& _,
                                      const mlir::Value& to) const {
  throw IssueError(19);
}

mlir::Value CNamedType::GenDefault(mlir::MLIRContext* C,
                                   const mlir::Location& loc,
                                   mlir::OpBuilder& builder,
                                   const mlir::Value& to) const {
  assert(false && "CNamedType is only used in sourcegen.");
  return {};
}

}  // namespace pj
