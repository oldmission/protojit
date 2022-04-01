#pragma once

#include <filesystem>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include "context.hpp"
#include "protojit.hpp"
#include "types.hpp"

namespace pj {

using SourceId = std::vector<std::string>;

struct ParsedProtoFile {
  enum class DeclKind { kType, kComposite /*, kProtocol*/ };

  struct Decl {
    const DeclKind kind;
    const SourceId name;
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

struct SourceIdLess : std::less<SourceId> {
  using is_transparent = void;

  template <typename T, typename U>
  bool operator()(const T& a, const U& b) const {
    uintptr_t limit = std::min(a.size(), b.size());
    for (uintptr_t i = 0; i < limit; ++i) {
      if (a[i] < b[i]) {
        return true;
      } else if (b[i] < a[i]) {
        return false;
      }
    }
    return a.size() < b.size();
  }
};

struct ParsingScope {
  ProtoJitContext& ctx;
  std::map<std::filesystem::path, ParsedProtoFile> parsed_files;
  std::map<SourceId, mlir::Type, SourceIdLess> type_defs;
  // std::map<SourceId, mlir::Type, SourceIdLess> protocol_defs;

  std::set<std::filesystem::path> pending_files;
  std::vector<std::filesystem::path> stack;

  std::set<std::filesystem::path> import_dirs;
};

void ParseProtoFile(ParsingScope& scope, const std::filesystem::path&);

void GenerateHeader(const ArchDetails& arch, const ParsedProtoFile& file,
                    std::ostream& output);

}  // namespace pj
