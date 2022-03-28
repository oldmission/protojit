#pragma once

#include <filesystem>

#include "protojit.hpp"

namespace pj {

struct ParsedProtoFile {
  enum class DeclKind { kType, kComposite/*, kProtocol*/ };

  struct Decl {
    const DeclKind kind;
    const SourceId name;
    const AType* const atype;
    const CType* ctype;

    // For variants, whether the variant was declared as an enum.
    // We will generate a enum directly without the wrapper class
    // in this case.
    const bool is_enum = false;

    // For variants, explicitly defined tag values.
    //
    // Explicitly defined tag values are currently limited
    // to 1 byte.
    const std::map<std::string, uint8_t> explicit_tags;

    // For external structs, save the source order of the fields.
    const std::vector<std::string> field_order;

    bool is_external = false;
  };

  std::vector<Decl> decls;
  std::vector<std::filesystem::path> imports;
};

struct ParsingScope {
  Scope& scope;
  std::map<std::filesystem::path, ParsedProtoFile> parsed_files;
  std::map<SourceId, const ANamedType*> type_defs;
  // std::map<SourceId, const AType*> protocol_defs;

  std::set<std::filesystem::path> pending_files;
  std::vector<std::filesystem::path> stack;

  std::set<std::filesystem::path> import_dirs;
};

void ParseProtoFile(ParsingScope& scope, const std::filesystem::path&);

void PlanMemory(Scope* scope, ParsedProtoFile& file);

void GenerateHeader(Scope* scope, const ArchDetails& arch,
                    const ParsedProtoFile& file, std::ostream& output);

}  // namespace pj
