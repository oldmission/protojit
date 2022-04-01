#pragma once

#include <filesystem>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include "protojit.hpp"
#include "types.hpp"

namespace pj {

struct ParsedProtoFile {
  enum class DeclKind { kType, kComposite /*, kProtocol*/ };

  struct Decl {
    const DeclKind kind;
    const pj::types::Name name;
    mlir::Type type;

    // For variants, whether the variant was declared as an enum.
    // We will generate a enum directly without the wrapper class
    // in this case.
    const bool is_enum = false;

    bool is_external = false;
  };

  std::vector<Decl> decls;
  std::vector<std::filesystem::path> imports;
};

struct ParsingScope {
  Scope& scope;
  std::map<std::filesystem::path, ParsedProtoFile> parsed_files;
  std::map<pj::types::Name, mlir::Type> type_defs;
  // std::map<pj::types::Name, mlir::Type> protocol_defs;

  std::set<std::filesystem::path> pending_files;
  std::vector<std::filesystem::path> stack;

  std::set<std::filesystem::path> import_dirs;
};

void ParseProtoFile(ParsingScope& scope, const std::filesystem::path&);

void PlanMemory(Scope* scope, ParsedProtoFile& file);

void GenerateHeader(Scope* scope, const ArchDetails& arch,
                    const ParsedProtoFile& file, std::ostream& output);

}  // namespace pj
