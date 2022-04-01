#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "arch.hpp"
#include "context.hpp"
#include "exceptions.hpp"
#include "portal.hpp"
#include "protocol.hpp"
#include "tag.hpp"
#include "target.hpp"

namespace pj {

enum class Side {
  kSend,
  kRecv,
};

struct NegotiateOptions {
  bool allow_truncation = true;
};

const Protocol* PlanProtocol(ProtoJitContext* ctx, const ProtoSpec* spec);

const Protocol* Negotiate(ProtoJitContext* scope, const ProtoSpec* recv,
                          const ProtoSpec* send, NegotiateOptions opts);

std::unique_ptr<Portal> Compile(ProtoJitContext* ctx, const PortalSpec* spec);

template <typename T, size_t N, size_t M>
struct ArrayView {
  const uint64_t length;
  const intptr_t offset;

  ArrayView(const T* data, uint64_t length)
      : length(length),
        offset(reinterpret_cast<const char*>(data) -
               reinterpret_cast<const char*>(this)) {}
};

namespace gen {

template <typename T>
struct BuildPJType {};

}  // namespace gen

}  // namespace pj
