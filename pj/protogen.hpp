#pragma once

#include <filesystem>
#include <set>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include "types.hpp"

namespace pj {

using SourceId = std::vector<std::string>;

struct ParsedProtoFile {
  enum class DeclKind { kType, kComposite, kProtocol };

  struct Decl {
    const DeclKind kind;
    const SourceId name;
    mlir::Type type;

    // For variants, whether the variant was declared as an enum.
    // We will generate a enum directly without the wrapper class
    // in this case.
    const bool is_enum = false;

    // For protocol defs only.
    types::PathAttr path;

    bool is_external = false;
  };

  struct Portal {
    struct Sizer {
      std::string name;
      const SourceId src;
      types::PathAttr src_path;
      bool round_up;
    };

    struct Encoder {
      std::string name;
      const SourceId src;
      types::PathAttr src_path;
    };

    struct Decoder {
      std::string name;
      const SourceId dst;
      std::vector<types::PathAttr> handlers;
    };

    std::vector<Sizer> sizers;
    std::vector<Encoder> encoders;
    std::vector<Decoder> decoders;

    const std::string jit_class_name;

    // class name -> protocol
    std::map<std::string, SourceId> precomps;
  };

  std::vector<Decl> decls;
  std::map<SourceId, Portal> portals;
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
      }
      if (b[i] < a[i]) {
        return false;
      }
    }
    return a.size() < b.size();
  }
};

struct ParsingScope {
  mlir::MLIRContext& ctx;
  std::map<std::filesystem::path, ParsedProtoFile> parsed_files;
  std::map<SourceId, types::ValueType, SourceIdLess> type_defs;
  std::map<SourceId, std::pair<types::ValueType, types::PathAttr>, SourceIdLess>
      protocol_defs;

  std::set<std::filesystem::path> pending_files;
  std::vector<std::filesystem::path> stack;

  std::set<std::filesystem::path> import_dirs;
};

void parseProtoFile(ParsingScope& scope, const std::filesystem::path&);

void generateHeader(const ArchDetails& arch, const ParsedProtoFile& file,
                    std::ostream& output);

}  // namespace pj
