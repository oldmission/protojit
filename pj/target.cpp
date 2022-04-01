#include "target.hpp"

#include <memory>

// llvm
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCSymbol.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

// mlir
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include "ir.hpp"
#include "portal.hpp"
#include "protocol.hpp"

namespace pj {
Target::~Target() {}

using namespace ir;
using namespace mlir;

FuncOp SizeTarget::Compile(const ArchDetails& arch, Scope* S,
                           MLIRContext* C) const {
  OpBuilder _(C);

  auto t_from = mem->toIR(C);

  llvm::SmallVector<mlir::Type, 1> arg_types = {t_from};
  llvm::SmallVector<mlir::Type, 1> result_types = {mlir::IndexType::get(C)};

  auto loc = _.getUnknownLoc();
  auto func_type = _.getFunctionType(arg_types, result_types);
  auto func = mlir::FuncOp::create(loc, name, func_type);

  auto entryBlock = func.addEntryBlock();
  _.setInsertionPointToStart(entryBlock);

  auto size =
      mem->GenSize(S, C, path, proto->head, _, entryBlock->getArgument(0));

  const auto head_size = proto->head->total_size();
  if (head_size.IsNotNone()) {
    size = _.create<AddIOp>(loc, size,
                            GetIndexConstant(loc, _, head_size.bytes()));
  } else {
    // mem->GenSize is responsible for adding in self size when tags are used
  }

  size = _.create<AddIOp>(loc, size,
                          GetIndexConstant(loc, _, proto->tag_size_.bytes()));

  _.create<ReturnOp>(loc, size);

  return func;
}

FuncOp EncodeTarget::Compile(const ArchDetails& arch, Scope* S,
                             MLIRContext* C) const {
  OpBuilder _(C);

  auto t_from = mem->toIR(C);
  auto t_to = proto->head->toIR(C);

  llvm::SmallVector<mlir::Type, 2> arg_types = {t_from, t_to};

  auto loc = _.getUnknownLoc();
  auto func_type = _.getFunctionType(arg_types, llvm::None);

  auto func = mlir::FuncOp::create(loc, name, func_type);

  // Mark both arguments to the function as "noalias" to aid
  // LLVM optimization.
  func.setArgAttr(0, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(C));
  func.setArgAttr(1, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(C));

  auto entryBlock = func.addEntryBlock();
  _.setInsertionPointToStart(entryBlock);

  const Value tag =
      mem->GenEncode(S, C, proto->tag, path, proto->head, _,
                     entryBlock->getArgument(0), entryBlock->getArgument(1));

  if (!IsEmptyTag(proto->tag)) {
    assert(tag != nullptr);

    // SAMIR_TODO: Should size be passed in to encode() instead?
    const auto total_size = proto->SizeOf(mem, path);
    if (total_size.IsNone()) {
      throw IssueError(15);
    }

    _.create<ETagOp>(loc, entryBlock->getArgument(1), tag,
                     total_size - proto->tag_size_);
  }

  _.create<ReturnOp>(loc);

  return func;
}

FuncOp DecodeTarget::Compile(const ArchDetails& arch, Scope* S,
                             MLIRContext* C) const {
  OpBuilder _(C);

  auto t_from = proto->head->toIR(C);
  auto t_to = mem->toIR(C);
  auto t_size = GetIndexType(_);
  auto t_state = UserStateType::get(C);

  llvm::SmallVector<mlir::Type, 4> arg_types = {t_state, t_from, t_to, t_size};

  auto loc = _.getUnknownLoc();
  auto func_type = _.getFunctionType(arg_types, llvm::None);

  auto func = mlir::FuncOp::create(loc, name, func_type);

  // Mark both arguments to the function as "noalias" to aid
  // LLVM optimization.
  func.setArgAttr(1, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(C));
  func.setArgAttr(2, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                  UnitAttr::get(C));

  auto entryBlock = func.addEntryBlock();
  _.setInsertionPointToStart(entryBlock);

  Value tag{};
  if (!IsEmptyTag(proto->tag)) {
    // TODO(3): architecture-dependent
    // TODO: factor in joint tag alignment?

    // TODO: validate
    assert(proto->tag_size_.IsBytes());
    auto tag_offset =
        _.create<SubIOp>(loc, entryBlock->getArgument(3),
                         GetIndexConstant(loc, _, proto->tag_size_.bytes()));
    tag = _.create<LTagOp>(loc, _.getIntegerType(proto->tag_size_.bits(), true),
                           /*from=*/entryBlock->getArgument(1),
                           /*byte_offset=*/tag_offset,
                           /*width=*/proto->tag_size_);
  }

  proto->head->GenDecode(S, C, *this, proto->tag, dispatch_path, mem, _,
                         /*base=*/entryBlock->getArgument(2),
                         /*tag=*/tag,
                         /*from=*/entryBlock->getArgument(1),
                         /*to=*/entryBlock->getArgument(2),
                         /*user_state=*/entryBlock->getArgument(0));

  _.create<ReturnOp>(loc);

  return func;
}

}  // namespace pj
