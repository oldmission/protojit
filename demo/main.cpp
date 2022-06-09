#include <fstream>
#include <iostream>
#include <vector>

#include "pj/runtime.hpp"

#include "demo/adoption.v1.pj.hpp"
#include "demo/adoption.v2.pj.hpp"

namespace cl = llvm::cl;

cl::opt<std::string> SchemaFile(
    "schema", cl::desc("The schema file that will be used or written to"));

cl::opt<bool> UseVersion2("v2", cl::desc("Whether to use v2 or v1"));

void writeSchemaToFile(PJContext* ctx, const PJProtocol* proto,
                       const char* file) {
  std::vector<char> buf;
  buf.resize(pj::getProtoSize(ctx, proto));
  pj::encodeProto(ctx, proto, buf.data());

  std::ofstream{file}.write(buf.data(), buf.size());
}

const PJProtocol* readSchemaFromFile(PJContext* ctx, const char* file) {
  std::vector<char> buf;
  std::ifstream fs{file};

  size_t size = 0;
  do {
    buf.resize(size + 1024);
    fs.read(buf.data() + size, 1024);
    size += fs.gcount();
  } while (!fs.eof());

  return pj::decodeProto(ctx, buf.data());
}

int main(int argc, char** argv) {
  PJContext* ctx = pj::getContext();

  const PJProtocol* proto = pj::plan<v1::Adoption>(ctx);
  writeSchemaToFile(ctx, proto, argv[1]);
  proto = readSchemaFromFile(ctx, argv[1]);

  pj::getProtoSize(ctx, proto);

  pj::addSizeFunction<v1::Adoption>(ctx, "size", proto, /*src_path=*/"",
                                    /*round_up=*/true);
  pj::addEncodeFunction<v1::Adoption>(ctx, "encode", proto, /*src_path=*/"");
  pj::addDecodeFunction<v1::Adoption>(ctx, "decode", proto, /*handlers=*/{});

  auto portal = pj::compile(ctx);

  const auto size = portal->GetSizeFunction<v1::Adoption>("size");
  const auto encode = portal->GetEncodeFunction<v1::Adoption>("encode");
  const auto decode = portal->GetDecodeFunction<v1::Adoption>("decode");

  v1::Adoption adoption{
      .animal = {.specifics = {.value = {.cat = {.coat = v1::CatCoat::SHORT}},
                               .tag = v1::Specifics::Kind::cat},
                 .age = 25,
                 .weight = 10,
                 .sex = v1::Sex::MALE},
      .date = {.year = 2022, .month = 6, .date = 8},
      .fee = 0};

  std::vector<char> buf;
  buf.resize(size(&adoption));

  encode(&adoption, buf.data());

  v1::Adoption dst{.animal = {.specifics = {.value = {.cat = {}}}}};
  std::vector<char> dec_buf;
  dec_buf.resize(1024);

  decode(buf.data(), &dst, std::make_pair(dec_buf.data(), dec_buf.size()),
         nullptr);

  assert(dst.animal.specifics.tag == v1::Specifics::Kind::cat);
  assert(dst.animal.specifics.value.cat.coat == v1::CatCoat::SHORT);
  assert(dst.animal.age == 25);
  assert(dst.animal.weight == 10);
  assert(dst.animal.sex == v1::Sex::MALE);
  assert(dst.date.year == 2022);
  assert(dst.date.month == 6);
  assert(dst.date.date == 8);
  assert(dst.fee == 0);

  pj::freeContext(ctx);
}
