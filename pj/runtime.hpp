#pragma once

#include "protojit.hpp"
#include "runtime.h"

namespace pj {

class Portal;

PJContext* getContext();

void freeContext(PJContext* ctx);

template <typename Head>
const PJProtocol* plan(PJContext* ctx, const std::string& tag_path = "") {
  const void* head = gen::BuildPJType<Head>::build(ctx);
  return PJPlanProtocol(ctx, head, tag_path.c_str());
}

template <typename Proto>
const PJProtocol* planProtocol(PJContext* ctx) {
  using Head = typename gen::ProtocolHead<Proto>::Head;
  return plan<Head>(ctx, gen::ProtocolHead<Proto>::tag());
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

template <typename Src>
void addSizeFunction(PJContext* ctx, const std::string& name,
                     const PJProtocol* protocol, const std::string& src_path,
                     bool round_up) {
  PJAddSizeFunction(ctx, name.c_str(), gen::BuildPJType<Src>::build(ctx),
                    protocol, src_path.c_str(), round_up);
}

std::unique_ptr<Portal> compile(PJContext* ctx);

}  // namespace pj
