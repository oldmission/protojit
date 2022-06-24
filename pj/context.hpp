#pragma once

#include <memory>

#include <llvm/IR/Module.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

#include "portal.hpp"
#include "types.hpp"
#include "util.hpp"

namespace pj {

struct ProtoJitContext {
  ProtoJitContext();
  ~ProtoJitContext();

  uint64_t getProtoSize(types::ProtocolType proto);

  void encodeProto(types::ProtocolType proto, char* buf);

  types::ProtocolType decodeProto(const char* buf);

  void addEncodeFunction(std::string_view name, mlir::Type src,
                         types::ProtocolType protocol,
                         llvm::StringRef src_path);

  void addDecodeFunction(std::string_view name, types::ProtocolType protocol,
                         mlir::Type dst,
                         const std::vector<std::string>& handlers);

  // Exactly one of 'src' or 'dst' must be a protocol.
  void addSizeFunction(std::string_view name, mlir::Type src,
                       types::ProtocolType protocol, llvm::StringRef src_path,
                       bool round_up);

  void addProtocolDefinition(std::string_view name, std::string_view size_name,
                             llvm::StringRef proto_data);

  void precompile(std::string_view filename, size_t opt_level = 3);
  std::unique_ptr<Portal> compile(size_t opt_level = 3);

  // TODO: make these private after removing old compile API.
  mlir::MLIRContext ctx_;
  mlir::OpBuilder builder_;
  mlir::OwningModuleRef module_;

 private:
  void resetModule();

  // Decodes specifically the portion of a schema corresponding to the current
  // version.
  types::ProtocolType decodeProtoCurrentVersion(const char* buf);

  std::pair<std::unique_ptr<llvm::LLVMContext>, std::unique_ptr<llvm::Module>>
  compileToLLVM(size_t opt_level);

  DISALLOW_COPY_AND_ASSIGN(ProtoJitContext);
};

}  // namespace pj
