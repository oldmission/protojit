#include "context.hpp"
#include "ir.hpp"

namespace pj {

ProtoJitContext::ProtoJitContext()
    : builder_(&ctx_),
      module_(mlir::ModuleOp::create(builder_.getUnknownLoc())) {
  ctx_.getOrLoadDialect<pj::ir::ProtoJitDialect>();

  constexpr llvm::StringRef kUnitName = "<unit>";
  auto unit = pj::types::StructType::get(&ctx_, pj::types::TypeDomain::kHost,
                                         pj::Span<llvm::StringRef>{kUnitName});
  unit.setTypeData({
      .fields = pj::Span<pj::types::StructField>{nullptr, 0ul},
      .size = pj::Bytes(0),
      .alignment = pj::Bytes(0),
  });
  unit_type_ = unit;
}

ProtoJitContext::~ProtoJitContext() {}

}  // namespace pj
