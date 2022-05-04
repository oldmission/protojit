#pragma once

#include <memory>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include "params.hpp"
#include "passes.hpp"
#include "portal.hpp"
#include "types.hpp"
#include "util.hpp"

namespace pj {

struct ProtoJitContext {
  ProtoJitContext();
  ~ProtoJitContext();

  void addEncodeFunction(std::string_view name, mlir::Type src,
                         types::ProtocolType protocol,
                         llvm::StringRef src_path);

  void addDecodeFunction(
      std::string_view name, types::ProtocolType protocol, mlir::Type dst,
      const std::vector<std::pair<std::string, const void*>>& handlers);

  // Exactly one of 'src' or 'dst' must be a protocol.
  void addSizeFunction(std::string_view name, mlir::Type src,
                       types::ProtocolType protocol, llvm::StringRef src_path);

  std::unique_ptr<Portal> compile(const CompilationParams& params);

  pj::types::ValueType unitType() const { return unit_type_; }

  // TODO: make these private after removing old compile API.
  mlir::MLIRContext ctx_;
  mlir::OpBuilder builder_;
  mlir::OwningModuleRef module_;
  pj::types::ValueType unit_type_;

 private:
  DISALLOW_COPY_AND_ASSIGN(ProtoJitContext);
};

}  // namespace pj
