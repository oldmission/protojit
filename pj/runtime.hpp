#pragma once

#include <tuple>
#include <vector>

#include "portal_types.hpp"
#include "protojit.hpp"
#include "runtime.h"

namespace pj {

namespace runtime {

class Context;

class Portal {
 public:
  ~Portal() { PJFreePortal(portal_); }

  // Disable copying.
  Portal(const Portal&) = delete;
  Portal& operator=(const Portal&) = delete;

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
    return reinterpret_cast<DecodeFunction<T, ::pj::BoundedBuffer>>(
        PJGetDecodeFunction(portal_, name));
  }

 private:
  Portal(const PJPortal* portal) : portal_(portal) {}

  const PJPortal* portal_;
  friend class Context;
};

class Context {
 public:
  Context() : ctx_(PJGetContext()) {}
  ~Context() { PJFreeContext(ctx_); }

  // Disable copying.
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  template <typename Head>
  const PJProtocol* plan(const std::string& tag_path = "") {
    const void* head = gen::BuildPJType<Head>::build(ctx_);
    return PJPlanProtocol(ctx_, head, tag_path.c_str());
  }

  template <typename Proto>
  const PJProtocol* planProtocol() {
    using Head = typename gen::ProtocolHead<Proto>::Head;
    return plan<Head>(gen::ProtocolHead<Proto>::tag());
  }

  uint64_t getProtoSize(const PJProtocol* proto) {
    return PJGetProtoSize(ctx_, proto);
  }

  void encodeProto(const PJProtocol* proto, char* buf) {
    return PJEncodeProto(ctx_, proto, buf);
  }

  const PJProtocol* decodeProto(const char* buf) {
    return PJDecodeProto(ctx_, buf);
  }

  template <typename Src>
  void addEncodeFunction(const std::string& name, const PJProtocol* protocol,
                         const std::string& src_path) {
    PJAddEncodeFunction(ctx_, name.c_str(), gen::BuildPJType<Src>::build(ctx_),
                        protocol, src_path.c_str());
  }

  template <typename Dest>
  void addDecodeFunction(const std::string& name, const PJProtocol* protocol,
                         const std::vector<std::string>& handlers) {
    std::vector<const char*> handlers_arr;
    handlers_arr.reserve(handlers.size());
    for (const auto& name : handlers) {
      handlers_arr.push_back(name.c_str());
    }
    PJAddDecodeFunction(ctx_, name.c_str(), protocol,
                        gen::BuildPJType<Dest>::build(ctx_), handlers.size(),
                        handlers.empty() ? nullptr : &handlers_arr[0]);
  }

  template <typename Src>
  void addSizeFunction(const std::string& name, const PJProtocol* protocol,
                       const std::string& src_path, bool round_up) {
    PJAddSizeFunction(ctx_, name.c_str(), gen::BuildPJType<Src>::build(ctx_),
                      protocol, src_path.c_str(), round_up);
  }

  void precompile(const std::string& filename) {
    return PJPrecompile(ctx_, filename.c_str());
  }

  Portal compile() { return Portal(PJCompile(ctx_)); }

 private:
  PJContext* ctx_;
};

}  // namespace runtime
}  // namespace pj
