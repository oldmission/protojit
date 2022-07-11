#include <fstream>
#include <iostream>

#include "llvm/Support/CommandLine.h"

#include <pj/any.hpp>
#include <pj/runtime.hpp>

namespace cl = llvm::cl;

cl::OptionCategory Category("pjcat");

cl::opt<bool> PrintSchema("p", cl::desc("Print schema description"),
                          cl::cat(Category));

cl::opt<std::string> DataFile(cl::Positional, cl::desc("<data file>"),
                              cl::Required, cl::cat(Category));

void print(const pj::Any& any) {
  switch (any.kind()) {
    case pj::Any::Kind::Int: {
      pj::AnyInt in{any};
      if (in.sign() == pj::Sign::kSigned) {
        std::cout << in.getValue<int64_t>();
      } else {
        std::cout << in.getValue<uint64_t>();
      }
      break;
    }
    case pj::Any::Kind::Unit: {
      std::cout << "{}";
      break;
    }
    case pj::Any::Kind::Struct: {
      pj::AnyStruct str{any};
      std::cout << "{";
      bool first = true;
      for (const auto& field : str) {
        if (!first) {
          std::cout << ",";
        }
        first = false;
        std::cout << "\"" << field.name() << "\":";
        print(field.value());
      }
      std::cout << "}";
      break;
    }
    case pj::Any::Kind::Variant: {
      pj::AnyVariant var{any};
      std::cout << "{\"tag\":" << var.termTag()  //
                << ",\"name\":\"" << var.termName() << "\"";
      if (!var.isEnum()) {
        std::cout << ",\"data\":";
        print(var.term());
      }
      std::cout << "}";
      break;
    }
    case pj::Any::Kind::Sequence: {
      pj::AnySequence seq{any};
      if (seq.isString()) {
        std::cout << "\"";
        std::string str;
        for (const auto& val : seq) {
          str += pj::AnyInt{val}.getValue<char>();
        }
        std::cout << str << "\"";
      } else {
        std::cout << "[";
        bool first = true;
        for (const auto& val : seq) {
          if (!first) {
            std::cout << ",";
          }
          first = false;
          print(val);
        }
        std::cout << "]";
      }
      break;
    }
    default:
      std::cout << "\"undef\"";
      break;
  }
}

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(Category);
  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream fs{DataFile.getValue()};
  if (!fs) {
    std::cerr << "Failed to open file.\n";
    return 1;
  }

  // Get the protocol from the beginning of the data file.
  std::vector<char> proto_buf;
  proto_buf.resize(8);

  fs.read(proto_buf.data(), 8);
  if (fs.gcount() != 8) {
    std::cerr << "Invalid PJ data file" << std::endl;
    return 1;
  }

  int64_t proto_length;
  std::memcpy(&proto_length, proto_buf.data(), 8);

  proto_buf.resize(proto_length);
  fs.read(proto_buf.data() + 8, proto_length - 8);
  if (fs.gcount() != proto_length - 8) {
    std::cerr << "Invalid PJ data file" << std::endl;
    return 1;
  }

  pj::runtime::Context ctx;
  auto proto = ctx.decodeProto(proto_buf.data());

  if (PrintSchema.getValue()) {
    proto.printLayout();
  }

  ctx.addDecodeFunction<pj::Any>("decode", proto, {});
  auto portal = ctx.compile();

  const auto decode = portal.getDecodeFunction<pj::Any>("decode");

  pj::Any any;
  int64_t dec_size = 1024;
  auto dec_buffer = std::make_unique<char[]>(dec_size);
  while (fs.peek() != std::char_traits<char>::eof()) {
    int64_t msg_length;
    fs.read(reinterpret_cast<char*>(&msg_length), 8);
    if (fs.gcount() != 8) {
      std::cerr << "Invalid PJ data file" << std::endl;
      return 1;
    }

    std::vector<char> msg_buf;
    msg_buf.resize(msg_length);
    fs.read(msg_buf.data(), msg_length);
    if (fs.gcount() != msg_length) {
      std::cerr << "Invalid PJ data file" << std::endl;
      return 1;
    }

    while (true) {
      auto bbuf =
          decode(msg_buf.data(), &any,
                 {.ptr = dec_buffer.get(), .size = dec_size}, nullptr, nullptr);

      if (bbuf.size >= 0) {
        print(any);
        std::cout << std::endl;
        break;
      }

      dec_size *= 2;
      auto dec_buffer = std::make_unique<char[]>(dec_size);
    }
  }

  return 0;
}
