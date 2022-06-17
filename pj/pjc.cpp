#include <filesystem>
#include <fstream>
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
    cl::desc("Generate an optimized wire protocol for the specified protocol"),
    cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> OuterNamespace(
    "space",
    cl::desc("Puts all the generated files in the specified namespace"),
    cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> InputFile(cl::Positional, cl::desc("<proto file>"),
                               cl::Required, cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> OutputHeader("hpp", cl::desc("output header file"),
                                  cl::Required,
                                  cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> OutputTU("cpp", cl::desc("output cpp file (precomp only)"),
                              cl::cat(ProtogenOptionsCategory));

pj::SourceId decodeScopedName(const std::string& str) {
  pj::SourceId name;
  if (str.empty()) {
    return name;
  }

  size_t pos = 0;
  do {
    auto next = str.find('.', pos);
    name.push_back(str.substr(pos, next - pos));
    pos = next + 1;
    if (next == std::string::npos) {
      break;
    }
  } while (true);
  return name;
}

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(ProtogenOptionsCategory);
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<pj::ir::ProtoJitDialect>();

  pj::ParsingScope parse_scope{.ctx = ctx};

  const auto input_path = std::filesystem::path(InputFile.getValue());
  const auto header_path = std::filesystem::path(OutputHeader.getValue());

  for (auto& dir : IncludeDirectories) {
    parse_scope.import_dirs.emplace(dir);
  }

  try {
    pj::parseProtoFile(parse_scope, input_path);
  } catch (tao::pegtl::parse_error& e) {
    std::cerr << "Parse error: " << e.what() << "\n";
    return 1;
  }

  auto& parsed_file = parse_scope.parsed_files.at(input_path);

  pj::SourceGenerator sourcegen{decodeScopedName(OuterNamespace.getValue())};

  const auto& proto = GenerateProtocol.getValue();
  if (!proto.empty()) {
    auto name = decodeScopedName(proto);
    auto it = parse_scope.protocol_defs.find(name);
    if (it == parse_scope.protocol_defs.end()) {
      std::cerr << "Protocol provided with option --gen-proto not found.\n";
      return 1;
    }

    auto planned = pj::plan_protocol(ctx, it->second.first, it->second.second);
    // TODO: update this
    // sourcegen.addProtocol(name, planned);
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
          sourcegen.addProtocol(decl.name, decl.type, decl.path);
          break;
      }
    }

    for (auto& [ns, portal] : parsed_file.portals) {
      sourcegen.addPortal(ns, portal);
    }
  }

  std::ofstream header_out{header_path};
  std::unique_ptr<std::ofstream> cpp_out;
  if (!OutputTU.empty()) {
    cpp_out.reset(new std::ofstream{OutputTU.getValue()});
  }
  sourcegen.generate(header_path, header_out, cpp_out.get(),
                     parsed_file.imports);
  return 0;
}
