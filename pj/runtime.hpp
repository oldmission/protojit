#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "protojit.hpp"
#include "runtime.h"

namespace pj {

namespace runtime {

class Context;

class Portal {
 public:
  Portal() : portal_(nullptr) {}
  ~Portal() { PJFreePortal(portal_); }

  // Disable copying.
  Portal& operator=(const Portal&) = delete;
  Portal(const Portal&) = delete;

  // Allow moving.
  Portal& operator=(Portal&& p) {
    portal_ = p.portal_;
    p.portal_ = nullptr;
    return *this;
  }
  Portal(Portal&& p) { *this = std::move(p); }

  template <typename T>
  auto getSizeFunction(const char* name) const {
    return reinterpret_cast<SizeFunction<T>>(PJGetSizeFunction(portal_, name));
  }

  template <typename T>
  auto getEncodeFunction(const char* name) const {
    return reinterpret_cast<EncodeFunction<T>>(
        PJGetEncodeFunction(portal_, name));
  }

  template <typename T>
  auto getDecodeFunction(const char* name) const {
    return reinterpret_cast<DecodeFunction<T, void>>(
        PJGetDecodeFunction(portal_, name));
  }

 private:
  Portal(const PJPortal* portal) : portal_(portal) {}

  const PJPortal* portal_;
  friend class Context;
};

class Protocol {
 public:
  Protocol(const Protocol& proto) : proto_(proto.proto_) {}

  bool isBinaryCompatibleWith(Protocol other) const {
    return PJIsBinaryCompatible(proto_, other.proto_);
  }

 private:
  Protocol(const PJProtocol* proto) : proto_(proto) {}

  const PJProtocol* proto_;
  friend class Context;
};

class Context {
 public:
  Context() : ctx_(PJGetContext()) {}
  ~Context() { PJFreeContext(ctx_); }

  // Disable copying.
  Context& operator=(const Context&) = delete;
  Context(const Context&) = delete;

  // Allow moving.
  Context& operator=(Context&& c) {
    ctx_ = c.ctx_;
    c.ctx_ = nullptr;
    return *this;
  }
  Context(Context&& c) { *this = std::move(c); }

  template <typename Head>
  Protocol plan(const std::string& tag_path = "") {
    const void* head =
        gen::BuildPJType<Head>::build(ctx_, PJGetHostDomain(ctx_));
    return Protocol{PJPlanProtocol(ctx_, head, tag_path.c_str())};
  }

  uint64_t getProtoSize(Protocol proto) {
    return PJGetProtoSize(ctx_, proto.proto_);
  }

  void encodeProto(Protocol proto, char* buf) {
    return PJEncodeProto(ctx_, proto.proto_, buf);
  }

  Protocol decodeProto(const char* buf) { return PJDecodeProto(ctx_, buf); }

  template <typename Src>
  void addEncodeFunction(const std::string& name, Protocol protocol,
                         const std::string& src_path) {
    PJAddEncodeFunction(
        ctx_, name.c_str(),
        gen::BuildPJType<Src>::build(ctx_, PJGetHostDomain(ctx_)),
        protocol.proto_, src_path.c_str());
  }

  template <typename Dest>
  void addDecodeFunction(const std::string& name, Protocol protocol,
                         const std::vector<std::string>& handlers) {
    std::vector<const char*> handlers_arr;
    handlers_arr.reserve(handlers.size());
    for (const auto& name : handlers) {
      handlers_arr.push_back(name.c_str());
    }
    PJAddDecodeFunction(
        ctx_, name.c_str(), protocol.proto_,
        gen::BuildPJType<Dest>::build(ctx_, PJGetHostDomain(ctx_)),
        handlers.size(), handlers.empty() ? nullptr : &handlers_arr[0]);
  }

  template <typename Src>
  void addSizeFunction(const std::string& name, Protocol protocol,
                       const std::string& src_path, bool round_up) {
    PJAddSizeFunction(ctx_, name.c_str(),
                      gen::BuildPJType<Src>::build(ctx_, PJGetHostDomain(ctx_)),
                      protocol.proto_, src_path.c_str(), round_up);
  }

  void addProtocolDefinition(const std::string& name,
                             const std::string& size_name, Protocol protocol) {
    PJAddProtocolDefinition(ctx_, name.c_str(), size_name.c_str(),
                            protocol.proto_);
  }

  void precompile(const std::string& filename, bool pic = false) {
    return PJPrecompile(ctx_, filename.c_str(), pic);
  }

  Portal compile() { return Portal(PJCompile(ctx_)); }

 private:
  PJContext* ctx_;
};

}  // namespace runtime
}  // namespace pj
