#include <csignal>
#include <fstream>
#include <iostream>
#include <vector>

#include "llvm/Support/CommandLine.h"

#include "pj/runtime.hpp"

#include "demo/adoption.pj.hpp"

namespace cl = llvm::cl;

enum class RunMode { kRead, kWrite };
enum class Ver { v1, v2 };

cl::OptionCategory DemoOptionsCategory("Protocol JIT Demo");

cl::opt<RunMode> Mode(
    "mode",
    cl::values(clEnumValN(RunMode::kRead, "read", "Read from existing file"),
               clEnumValN(RunMode::kWrite, "write", "Write to a new file")),
    cl::Required, cl::cat(DemoOptionsCategory));

cl::opt<Ver> Version(
    "proto-version", cl::desc("Which protocol version to use"),
    cl::values(clEnumValN(Ver::v1, "v1", "Version 1 of the protocol"),
               clEnumValN(Ver::v2, "v2", "Version 2 of the protocol")),
    cl::Required, cl::cat(DemoOptionsCategory));

cl::opt<std::string> SchemaFile(
    "schema", cl::desc("The schema file that will be used or written to"),
    cl::Required, cl::cat(DemoOptionsCategory));

cl::opt<std::string> DataFile(
    "file", cl::desc("The file data will be read or written from"),
    cl::Required, cl::cat(DemoOptionsCategory));

std::string CatName = "Onyx";
std::string DogName = "Toast";

v1::Adoption SampleCatV1{
    .animal = {.specifics = {.value = {.cat = {.coat = v1::CatCoat::SHORT}},
                             .tag = v1::Specifics::Kind::cat},
               .name = v1::Name{CatName.data(), CatName.length()},
               .age = 24,
               .weight = 7,
               .sex = v1::Sex::MALE},
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

v1::Adoption SampleDogV1{
    .animal = {.specifics = {.value = {.dog = {.breed =
                                                   std::array{
                                                       v1::DogBreed::BEAGLE}}},
                             .tag = v1::Specifics::Kind::dog},
               .name = v1::Name{DogName.data(), DogName.length()},
               .age = 36,
               .weight = 45,
               .sex = v1::Sex::FEMALE},
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

v2::Adoption SampleCatV2{
    .animal = {.specifics = {.value = {.cat = {.personality =
                                                   v2::CatPersonality::NERVOUS,
                                               .coat = v2::CatCoat::SHORT}},
                             .tag = v2::Specifics::Kind::cat},
               .name = v2::Name{CatName.data(), CatName.length()},
               .age = 24,
               .weight = 7,
               .sex = v2::Sex::MALE},
    .location = v2::Location::SHELTER_A,
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

v2::Adoption SampleDogV2{
    .animal = {.specifics =
                   {.value = {.dog = {.coat = v2::DogCoat::SHORT,
                                      .breed = std::array{v2::DogBreed::BEAGLE,
                                                          v2::DogBreed::HUSKY}}},
                    .tag = v2::Specifics::Kind::dog},
               .name = v2::Name{DogName.data(), DogName.length()},
               .age = 36,
               .weight = 45,
               .sex = v2::Sex::FEMALE},
    .location = v2::Location::SHELTER_C,
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

void writeSchemaToFile(pj::runtime::Context& ctx, pj::runtime::Protocol proto,
                       const std::string& file) {
  std::vector<char> buf;
  buf.resize(ctx.getProtoSize(proto));
  ctx.encodeProto(proto, buf.data());

  std::ofstream{file}.write(buf.data(), buf.size());
}

pj::runtime::Protocol readSchemaFromFile(pj::runtime::Context& ctx,
                                         const std::string& file) {
  std::vector<char> buf;
  std::ifstream fs{file};

  size_t size = 0;
  do {
    buf.resize(size + 1024);
    fs.read(buf.data() + size, 1024);
    size += fs.gcount();
  } while (!fs.eof());

  return ctx.decodeProto(buf.data());
}

template <Ver V>
using Protocol =
    std::conditional_t<V == Ver::v1, v1::AdoptionProto, v2::AdoptionProto>;

template <Ver V>
void read(pj::runtime::Context& ctx) {
  using Proto = Protocol<V>;
  using Head = typename pj::gen::ProtocolHead<Proto>::Head;

  auto proto = readSchemaFromFile(ctx, SchemaFile.getValue());
  std::cout << "Read optimized protocol from schema file" << std::endl;

  void (*handle_cat)(const Head*, const void*) = [](const Head* adoption,
                                                    const void*) {
    std::cout << "Got cat adoption message" << std::endl;
  };
  void (*handle_dog)(const Head*, const void*) = [](const Head* adoption,
                                                    const void*) {
    std::cout << "Got dog adoption message" << std::endl;
  };

  ctx.addDecodeFunction<Head>("decode", proto, /*handlers=*/
                              {"animal.specifics.cat", "animal.specifics.dog"});

  auto portal = ctx.compile();
  std::cout << "Compiled decode function" << std::endl;

  const auto decode = portal.getDecodeFunction<Head>("decode");

  std::vector<char> data_buf;
  data_buf.resize(1024);
  std::vector<char> dec_buf;
  dec_buf.resize(1024);
  std::ifstream fs{DataFile.getValue()};

  do {
    // Extract one message from the file.
    fs.read(data_buf.data(), 8);
    assert(fs.gcount() == 8);

    uint64_t length = *reinterpret_cast<uint64_t*>(data_buf.data());
    if (8 + length > data_buf.size()) data_buf.resize(8 + length);
    fs.read(data_buf.data() + 8, length);
    assert(fs.gcount() == length);

    // Decode the message, increasing the size of the decode buffer if
    // necessary.
    Head dst{.animal = {.specifics = {.value = {.cat = {}}}}};

    using HandlerT = void (*)(const Head*, const void*);
    HandlerT handlers[2] = {handle_cat, handle_dog};

    while (true) {
      auto bbuf = decode(data_buf.data() + 8, &dst,
                         {.ptr = dec_buf.data(), .size = dec_buf.size()},
                         handlers, nullptr);
      if (bbuf.ptr != nullptr) break;
      dec_buf.resize(dec_buf.size() * 2);
    }
  } while (fs.peek() != std::char_traits<char>::eof());
}

template <Ver V>
void write(pj::runtime::Context& ctx) {
  using Proto = Protocol<V>;
  using Head = typename pj::gen::ProtocolHead<Proto>::Head;

  auto proto = ctx.planProtocol<Proto>();
  std::cout << "Planned optimized protocol" << std::endl;
  writeSchemaToFile(ctx, proto, SchemaFile.getValue());
  std::cout << "Outputted schema to file" << std::endl;

  // Specialized functions generated for each of cat and dog because it is known
  // at compile-time which of the two we'll be encoding.
  ctx.addSizeFunction<Head>("size_cat", proto,
                            /*src_path=*/"animal.specifics.cat",
                            /*round_up=*/false);
  ctx.addSizeFunction<Head>("size_dog", proto,
                            /*src_path=*/"animal.specifics.dog",
                            /*round_up=*/false);
  ctx.addEncodeFunction<Head>("encode_cat", proto,
                              /*src_path=*/"animal.specifics.cat");
  ctx.addEncodeFunction<Head>("encode_dog", proto,
                              /*src_path=*/"animal.specifics.dog");

  auto portal = ctx.compile();
  std::cout << "Compiled size and encode functions" << std::endl;

  const auto size_cat = portal.getSizeFunction<Head>("size_cat");
  const auto size_dog = portal.getSizeFunction<Head>("size_dog");
  const auto encode_cat = portal.getEncodeFunction<Head>("encode_cat");
  const auto encode_dog = portal.getEncodeFunction<Head>("encode_dog");

  auto [cat, dog] = [&]() {
    if constexpr (V == Ver::v1) {
      return std::make_pair(&SampleCatV1, &SampleDogV1);
    } else {
      return std::make_pair(&SampleCatV2, &SampleDogV2);
    }
  }();

  uint64_t cat_size = size_cat(cat);
  uint64_t dog_size = size_dog(dog);
  uint64_t cat_size_aligned = ((cat_size - 1) | 7) + 1;

  std::vector<char> buf;
  buf.resize(8 + cat_size_aligned + 8 + dog_size);

  *reinterpret_cast<uint64_t*>(&buf[0]) = cat_size_aligned;
  encode_cat(cat, &buf[8]);

  *reinterpret_cast<uint64_t*>(&buf[8 + cat_size_aligned]) = dog_size;
  encode_dog(dog, &buf[8 + cat_size_aligned + 8]);

  std::ofstream{DataFile.getValue()}.write(buf.data(), buf.size());
  std::cout << "Encoded and outputted data to file" << std::endl;
}

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(DemoOptionsCategory);
  cl::ParseCommandLineOptions(argc, argv);

  pj::runtime::Context ctx;

  if (Mode.getValue() == RunMode::kRead) {
    if (Version.getValue() == Ver::v1) {
      read<Ver::v1>(ctx);
    } else {
      read<Ver::v2>(ctx);
    }
  } else {
    if (Version.getValue() == Ver::v1) {
      write<Ver::v1>(ctx);
    } else {
      write<Ver::v2>(ctx);
    }
  }
}
