#include <csignal>
#include <fstream>
#include <iostream>
#include <vector>

#include "llvm/Support/CommandLine.h"

#include "pj/runtime.hpp"

#include "demo/adoption.pj.hpp"

namespace cl = llvm::cl;

enum class RunMode { kRead, kWrite, kWriteExisting };
enum class Ver { v1, v2 };

cl::OptionCategory DemoOptionsCategory("Protocol JIT Demo");

cl::opt<RunMode> Mode(
    "mode",
    cl::values(clEnumValN(RunMode::kRead, "read", "Read from existing file"),
               clEnumValN(RunMode::kWrite, "write", "Write to a new file"),
               clEnumValN(RunMode::kWriteExisting, "writeexisting",
                          "Write to a new file, using an existing schema")),
    cl::Required, cl::cat(DemoOptionsCategory));

cl::opt<Ver> Version(
    "proto-version", cl::desc("Which protocol version to use"),
    cl::values(clEnumValN(Ver::v1, "v1", "Version 1 of the protocol"),
               clEnumValN(Ver::v2, "v2", "Version 2 of the protocol")),
    cl::Required, cl::cat(DemoOptionsCategory));

cl::opt<std::string> File(
    "file", cl::desc("The file that will be read from or written to"),
    cl::Required, cl::cat(DemoOptionsCategory));

cl::opt<std::string> SchemaFile(
    "schema",
    cl::desc(
        "The file that will be used as the schema for the writeexisting mode"),
    cl::Optional, cl::cat(DemoOptionsCategory));

std::string CatName = "Onyx";
std::string DogName = "Toast";

v1::Adoption SampleCatV1{
    .animal = {.specifics = {v1::Specifics::cat,
                             v1::Cat{.coat = v1::CatCoat::SHORT}},
               .name = v1::Name{CatName.data(), CatName.length()},
               .age = 24.5,
               .weight = 7,
               .sex = v1::Sex::MALE},
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

auto kBeagleBreed = std::array{v1::DogBreed::BEAGLE};
v1::Adoption SampleDogV1{
    .animal = {.specifics = {v1::Specifics::dog,
                             v1::Dog{.breed = {kBeagleBreed.data(),
                                               kBeagleBreed.size()},
                                     .short_int = 42}},
               .name = v1::Name{DogName.data(), DogName.length()},
               .age = 36.2,
               .weight = 45,
               .sex = v1::Sex::FEMALE},
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

v2::Adoption SampleCatV2{
    .animal = {.specifics = {v2::Specifics::cat,
                             v2::Cat{.personality = v2::CatPersonality::NERVOUS,
                                     .coat = v2::CatCoat::SHORT}},
               .name = v2::Name{CatName.data(), CatName.length()},
               .age = 24.5,
               .weight = 7,
               .sex = v2::Sex::MALE},
    .location = v2::Location::SHELTER_A,
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

auto kTwoBreeds = std::array{v2::DogBreed::BEAGLE, v2::DogBreed::HUSKY};
v2::Adoption SampleDogV2{
    .animal = {.specifics = {v2::Specifics::dog,
                             v2::Dog{.coat = v2::DogCoat::SHORT,
                                     .breed = {kTwoBreeds.data(),
                                               kTwoBreeds.size()}}},
               .name = v2::Name{DogName.data(), DogName.length()},
               .age = 36.2,
               .weight = 45,
               .sex = v2::Sex::FEMALE},
    .location = v2::Location::SHELTER_C,
    .date = {.year = 2022, .month = 6, .date = 9},
    .fee = 100,
};

template <Ver V>
using Reader = std::conditional_t<V == Ver::v1, v1::Reader, v2::Reader>;
template <Ver V>
using Writer = std::conditional_t<V == Ver::v1, v1::Writer, v2::Writer>;
template <Ver V>
using Adoption = std::conditional_t<V == Ver::v1, v1::Adoption, v2::Adoption>;

void writeSchemaToFile(pj::runtime::Context& ctx, pj::runtime::Protocol proto,
                       std::ofstream& fs) {
  std::vector<char> buf;
  buf.resize(ctx.getProtoSize(proto));
  ctx.encodeProto(proto, buf.data());

  fs.write(buf.data(), buf.size());
}

std::vector<char> readSchema(std::ifstream& fs) {
  if (!fs.is_open()) {
    std::cerr << "File does not exist!" << std::endl;
    exit(1);
  }

  std::vector<char> buf;
  buf.resize(8);

  fs.read(buf.data(), 8);
  assert(fs.gcount() == 8);

  int64_t length;
  std::memcpy(&length, buf.data(), 8);

  buf.resize(length);
  fs.read(buf.data() + 8, length - 8);
  assert(fs.gcount() == length - 8);

  return buf;
}

template <Ver V>
void read(pj::runtime::Context& ctx) {
  std::ifstream fs{File.getValue()};

  auto reader = Reader<V>{readSchema(fs).data()};
  reader.getProtocol().printLayout();

  auto handle_cat = [](const Adoption<V>* adoption, void*) {
    std::cout << "Got cat adoption message" << std::endl;
  };
  auto handle_dog = [](const Adoption<V>* adoption, void*) {
    std::cout << "Got dog adoption message" << std::endl;
    if constexpr (V == Ver::v1) {
      std::cout << "Got short_int: "
                << adoption->animal.specifics.value.dog.short_int << std::endl;
    }
  };

  std::vector<char> data_buf;
  data_buf.resize(1024);
  std::vector<char> dec_buf;
  dec_buf.resize(1024);

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
    Adoption<V> dst;

    auto handlers = std::array<pj::DecodeHandler<Adoption<V>, void>, 2>{
        handle_cat, handle_dog};

    for (int64_t dec_size = 1024;; dec_size *= 2) {
      auto dec_buf = std::make_unique<char[]>(dec_size);

      auto bbuf = reader.template decode<void>(
          data_buf.data() + 8, &dst, {.ptr = dec_buf.get(), .size = dec_size},
          handlers, nullptr);

      if (bbuf.size >= 0) break;
    }
  } while (fs.peek() != std::char_traits<char>::eof());
}

template <Ver V>
void write(pj::runtime::Context& ctx) {
  auto writer = [&]() {
    if (Mode.getValue() == RunMode::kWrite) {
      return Writer<V>{};
    } else {
      std::ifstream fs{SchemaFile.getValue()};
      return Writer<V>{readSchema(fs).data()};
    }
  }();

  std::ofstream fs{File.getValue()};
  writeSchemaToFile(ctx, writer.getProtocol(), fs);

  auto [cat, dog] = [&]() {
    if constexpr (V == Ver::v1) {
      return std::make_pair(&SampleCatV1, &SampleDogV1);
    } else {
      return std::make_pair(&SampleCatV2, &SampleDogV2);
    }
  }();

  uint64_t cat_size = writer.size_cat(cat);
  uint64_t dog_size = writer.size_dog(dog);
  uint64_t cat_size_aligned = ((cat_size - 1) | 7) + 1;

  std::vector<char> buf;
  buf.resize(8 + cat_size_aligned + 8 + dog_size);

  *reinterpret_cast<uint64_t*>(&buf[0]) = cat_size_aligned;
  writer.encode_cat(cat, &buf[8]);

  *reinterpret_cast<uint64_t*>(&buf[8 + cat_size_aligned]) = dog_size;
  writer.encode_dog(dog, &buf[8 + cat_size_aligned + 8]);

  fs.write(buf.data(), buf.size());
}

int main(int argc, char** argv) {
  cl::HideUnrelatedOptions(DemoOptionsCategory);
  cl::ParseCommandLineOptions(argc, argv);

  if (!SchemaFile.getValue().empty() &&
      Mode.getValue() != RunMode::kWriteExisting) {
    std::cerr
        << "--schema option should only be specified in writeexisting mode"
        << std::endl;
    return 1;
  } else if (SchemaFile.getValue().empty() &&
             Mode.getValue() == RunMode::kWriteExisting) {
    std::cerr << "--schema option must be provided in writeexisting mode"
              << std::endl;
    return 1;
  }

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
