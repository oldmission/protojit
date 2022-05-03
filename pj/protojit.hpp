#pragma once

#include <array>
#include <memory>
#include <optional>

#include "arch.hpp"
#include "exceptions.hpp"
#include "portal.hpp"
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

#if 0
const Protocol* Negotiate(PJContext* scope, const ProtoSpec* recv,
                          const ProtoSpec* send, NegotiateOptions opts);
#endif

template <typename T, size_t MinLength, intptr_t MaxLength>
class ArrayView {
 public:
  ArrayView(const T* data, uint64_t length)
      : length(length), outline(reinterpret_cast<const char*>(data)) {
    assert(MaxLength < 0 || length <= static_cast<uint64_t>(MaxLength));
    if (length <= MinLength) {
      std::copy(data, data + length, storage.begin());
    }
  }

  template <size_t N, typename = std::enable_if_t<N <= MinLength>>
  ArrayView(const std::array<T, N>& arr) : length(N), outline(nullptr) {
    std::copy(arr.begin(), arr.end(), storage.begin());
  }

  ArrayView() : length(0) {}

  ArrayView& operator=(const ArrayView& o) {
    length = o.length;
    outline = o.outline;
    if (length <= MinLength) {
      storage = o.storage;
    }
    return *this;
  }

  const T& operator[](uintptr_t i) const {
    if (length <= MinLength) {
      return storage[i];
    } else {
      return reinterpret_cast<const T*>(outline)[i];
    }
  }

  template <typename U>
  bool operator==(const U& o) const {
    return std::equal(begin(), end(), o.begin(), o.end());
  }

  const T* begin() const {
    if (length <= MinLength) {
      return storage.begin();
    } else {
      return reinterpret_cast<const T*>(outline);
    }
  }

  const T* end() const {
    if (length <= MinLength) {
      return storage.begin() + length;
    } else {
      return reinterpret_cast<const T*>(outline) + length;
    }
  }

  uint64_t size() const { return length; }

  bool has_ref() const { return length > MinLength; }

 private:
  uint64_t length;
  const char* outline;
  std::array<T, MinLength> storage;

  template <typename U>
  friend struct gen::BuildPJType;
};

template <typename T, intptr_t MaxLength>
class ArrayView<T, 0, MaxLength> {
 public:
  ArrayView(const T* data, uint64_t length)
      : length(length), outline(reinterpret_cast<const char*>(data)) {
    assert(MaxLength < 0 || length <= static_cast<uint64_t>(MaxLength));
  }

  ArrayView() : length(0) {}

  ArrayView& operator=(const ArrayView& o) {
    length = o.length;
    outline = o.outline;
    return *this;
  }

  const T& operator[](uintptr_t i) const {
    return reinterpret_cast<const T*>(outline)[i];
  }

  template <typename U>
  bool operator==(const U& o) const {
    return std::equal(begin(), end(), o.begin(), o.end());
  }

  const T* begin() const { return reinterpret_cast<const T*>(outline); }

  const T* end() const { return reinterpret_cast<const T*>(outline) + length; }

  uint64_t size() const { return length; }

 private:
  uint64_t length;
  const char* outline;

  template <typename U>
  friend struct gen::BuildPJType;
};

}  // namespace pj
