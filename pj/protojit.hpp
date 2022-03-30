#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "arch.hpp"
#include "exceptions.hpp"
#include "portal.hpp"
#include "protocol.hpp"
#include "scope.hpp"
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

const Protocol* PlanProtocol(Scope* scope, const ProtoSpec* spec);

const Protocol* Negotiate(  //
    Scope* scope, const ProtoSpec* recv, const ProtoSpec* send,
    NegotiateOptions opts);

void Validate(const Protocol* proto, const ProtoSpec* spec, Side side,
              NegotiateOptions opts);

std::unique_ptr<Portal> Compile(Scope* scope, const PortalSpec* spec);

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
