#ifndef PROTOJIT_PROTOJIT_HPP
#define PROTOJIT_PROTOJIT_HPP

#include <array>
#include <memory>
#include <optional>

#include "arch.hpp"
#include "exceptions.hpp"
#include "runtime.h"

namespace pj {

class Portal;

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

// SAMIR_TODO2: do we need to keep this?
#if 0
  ArrayView& operator=(const ArrayView& o) {
    length = o.length;
    outline = o.outline;
    if (length <= MinLength) {
      storage = o.storage;
    }
    return *this;
  }
#endif

  const T& operator[](uintptr_t i) const {
    if (length <= MinLength) {
      return storage[i];
    }
    return reinterpret_cast<const T*>(outline)[i];
  }

  template <typename U>
  bool operator==(const U& o) const {
    return std::equal(begin(), end(), o.begin(), o.end());
  }

  const T* begin() const {
    if (length <= MinLength) {
      return storage.begin();
    }
    return reinterpret_cast<const T*>(outline);
  }

  const T* end() const {
    if (length <= MinLength) {
      return storage.begin() + length;
    }
    return reinterpret_cast<const T*>(outline) + length;
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

// SAMIR_TODO2: do we need to keep this?
#if 0
  ArrayView& operator=(const ArrayView& o) {
    length = o.length;
    outline = o.outline;
    return *this;
  }
#endif

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

struct Any {
 private:
  const void* type_;
  const void* data_;

  template <typename U>
  friend struct gen::BuildPJType;
};

}  // namespace pj

#include "pj/reflect.pj.hpp"

namespace pj {
namespace gen {

template <>
struct BuildPJType<Any> {
  static const void* build(PJContext* ctx) {
    return PJCreateAnyType(
        ctx, offsetof(Any, data_) << 3, sizeof(Any::data_) << 3,
        offsetof(Any, type_) << 3, sizeof(Any::type_) << 3, sizeof(Any) << 3,
        alignof(Any) << 3,
        ::pj::gen::BuildPJType<::pj::reflect::Proto>::build(ctx));
  }
};

}  // namespace gen
}  // namespace pj

#endif  // PROTOJIT_PROTOJIT_HPP
