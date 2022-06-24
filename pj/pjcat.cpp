#include <fstream>
#include <iostream>

#include "llvm/Support/CommandLine.h"

#include <pj/runtime.hpp>

namespace cl = llvm::cl;

cl::OptionCategory Category("pjcat");

cl::opt<bool> PrintSchema("p", cl::desc("Print schema description"),
                          cl::cat(Category));

cl::opt<std::string> DataFile(cl::Positional, cl::desc("<data file>"),
                              cl::Required, cl::cat(Category));

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream fs{DataFile.getValue()};
  if (!fs) {
    std::cerr << "Failed to open file.\n";
    return 1;
  }

  if (PrintSchema.getValue()) {
    std::vector<char> buf;
    size_t size = 0;
    do {
      buf.resize(size + 1024);
      fs.read(buf.data() + size, 1024);
      size += fs.gcount();
    } while (!fs.eof());

    pj::runtime::Context ctx;
    auto protocol = ctx.decodeProto(buf.data());
    protocol.printLayout();
  } else {
    std::cerr << "Not yet implemented.\n";
    return 1;
  }

  return 0;
}
