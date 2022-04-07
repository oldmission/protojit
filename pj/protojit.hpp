#pragma once

#include <array>
#include <memory>
#include <optional>

#include "arch.hpp"
#include "exceptions.hpp"
#include "portal.hpp"
#include "protocol.hpp"
#include "runtime.h"

namespace pj {

namespace gen {

template <typename T>
struct BuildPJType {};

template <typename T>
struct ProtocolHead {};

}  // namespace gen

enum class Side {
  kSend,
  kRecv,
};

struct NegotiateOptions {
  bool allow_truncation = true;
};

PJContext* getContext();

void freeContext(PJContext* ctx);

template <typename Head>
const PJProtocol* plan(PJContext* ctx, const std::string& tag_path = "") {
  const void* head = gen::BuildPJType<Head>::build(ctx);
  return PJPlanProtocol(ctx, head, tag_path.c_str());
}

template <typename Proto>
const PJProtocol* planProtocol(PJContext* ctx,
                               const std::string& tag_path = "") {
  using Head = typename gen::ProtocolHead<Proto>::Head;
  return plan<Head>(ctx);
}

template <typename Src>
void addEncodeFunction(PJContext* ctx, const std::string& name,
                       const PJProtocol* protocol,
                       const std::string& src_path) {
  PJAddEncodeFunction(ctx, name.c_str(), gen::BuildPJType<Src>::build(ctx),
                      protocol, src_path.c_str());
}

template <typename Dest>
void addDecodeFunction(
    PJContext* ctx, const std::string& name, const PJProtocol* protocol,
    const std::vector<std::pair<std::string, const void*>>& handlers) {
  std::vector<PJHandler> storage;
  std::vector<const PJHandler*> handlers_arr;
  storage.reserve(handlers.size());
  for (const auto& [name, function] : handlers) {
    storage.push_back(PJHandler{.name = name.c_str(), .function = function});
    handlers_arr.push_back(&storage.back());
  }
  PJAddDecodeFunction(ctx, name.c_str(), protocol,
                      gen::BuildPJType<Dest>::build(ctx), handlers.size(),
                      handlers.empty() ? nullptr : &handlers_arr[0]);
}

std::unique_ptr<Portal> compile(PJContext* ctx);

#if 0
const Protocol* Negotiate(PJContext* scope, const ProtoSpec* recv,
                          const ProtoSpec* send, NegotiateOptions opts);
#endif

template <typename T>
struct ArrayView {
  const uint64_t length;
  const intptr_t offset;

  ArrayView(const T* data, uint64_t length)
      : length(length),
        offset(reinterpret_cast<const char*>(data) -
               reinterpret_cast<const char*>(this)) {}
};

}  // namespace pj
