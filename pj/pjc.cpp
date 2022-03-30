#include <filesystem>
#include <iostream>

#include <pegtl.hpp>

#include "llvm/Support/CommandLine.h"

#include "ir.hpp"
#include "protogen.hpp"

namespace cl = llvm::cl;

cl::OptionCategory ProtogenOptionsCategory("Protocol JIT Source Generator");

cl::list<std::string> IncludeDirectories("import-dir",
                                         cl::desc("Import directory"),
                                         cl::cat(ProtogenOptionsCategory));

cl::opt<std::string> InputFile(cl::Positional, cl::desc("<proto file>"),
                               cl::Required, cl::cat(ProtogenOptionsCategory));

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(ProtogenOptionsCategory);
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<pj::ir::ProtoJitDialect>();

  pj::Scope scope{&ctx};
  pj::ParsingScope parse_scope{.scope = scope};

  const auto path = std::filesystem::path(InputFile.getValue());

  for (auto& dir : IncludeDirectories) {
    parse_scope.import_dirs.emplace(dir);
  }

  try {
    pj::ParseProtoFile(parse_scope, path);
  } catch (tao::pegtl::parse_error& e) {
    std::cerr << "Parse error: " << e.what() << "\n";
    return 1;
  }

  auto& parsed_file = parse_scope.parsed_files.at(path);

  // TODO: support cross-compilation
  pj::GenerateHeader(&scope, pj::ArchDetails::Host(), parsed_file, std::cout);
  return 0;
}
