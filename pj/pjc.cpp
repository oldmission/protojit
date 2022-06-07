#include <filesystem>
#include <iostream>

#include <pegtl.hpp>

#include "llvm/Support/CommandLine.h"

#include "ir.hpp"
#include "plan.hpp"
#include "protogen.hpp"
#include "sourcegen.hpp"

namespace cl = llvm::cl;

cl::OptionCategory ProtogenOptionsCategory("Protocol JIT Source Generator");

cl::list<std::string> IncludeDirectories("import-dir",
                                         cl::desc("Import directory"),
                                         cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> GenerateProtocol(
    "gen-proto",
    cl::desc("Generate an optimized wire protocol for the specified protocol "
             "or type"),
    cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> InputFile(cl::Positional, cl::desc("<proto file>"),
                               cl::Required, cl::cat(ProtogenOptionsCategory));

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(ProtogenOptionsCategory);
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<pj::ir::ProtoJitDialect>();

  pj::ParsingScope parse_scope{.ctx = ctx};

  const auto path = std::filesystem::path(InputFile.getValue());

  for (auto& dir : IncludeDirectories) {
    parse_scope.import_dirs.emplace(dir);
  }

  try {
    pj::parseProtoFile(parse_scope, path);
  } catch (tao::pegtl::parse_error& e) {
    std::cerr << "Parse error: " << e.what() << "\n";
    return 1;
  }

  auto& parsed_file = parse_scope.parsed_files.at(path);

  // TODO: support cross-compilation
  pj::SourceGenerator sourcegen{pj::ArchDetails::Host()};

  const auto& proto = GenerateProtocol.getValue();
  if (!proto.empty()) {
    pj::SourceId name;
    size_t pos = 0;
    do {
      auto next = proto.find('.', pos);
      name.push_back(proto.substr(pos, next - pos));
      pos = next + 1;
      if (next == std::string::npos) {
        break;
      }
    } while (true);

    auto it = std::find_if(parsed_file.decls.begin(), parsed_file.decls.end(),
                           [&](const auto& decl) { return decl.name == name; });
    if (it == parsed_file.decls.end()) {
      std::cerr
          << "Protocol or type provided with option --gen-proto not found.\n";
      return 1;
    }

    const auto& decl = *it;
    auto planned = pj::plan_protocol(ctx, decl.type, decl.tag_path);
    sourcegen.addProtocol(name, planned);
  } else {
    for (auto& decl : parsed_file.decls) {
      auto type = decl.type.cast<pj::types::ValueType>();
      switch (decl.kind) {
        case pj::ParsedProtoFile::DeclKind::kType:
          sourcegen.addTypedef(decl.name, type);
          break;
        case pj::ParsedProtoFile::DeclKind::kComposite:
          sourcegen.addComposite(type, decl.is_external);
          break;
        case pj::ParsedProtoFile::DeclKind::kProtocol:
          sourcegen.addProtocolHead(decl.name, type, decl.tag_path);
          break;
      }
    }
  }

  sourcegen.generateHeader(std::cout, parsed_file.imports);
  return 0;
}
