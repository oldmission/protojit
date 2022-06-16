#pragma once

#include <tuple>

namespace pj {

struct ProtoJitContext;

struct Artifact {};

class Portal {
 public:
  using BoundedBuffer = std::pair<char*, uint64_t>;
  template <typename T>
  using SizeFunction = uintptr_t (*)(const T*);
  template <typename T>
  using EncodeFunction = void (*)(const T*, char*);
  template <typename T>
  using Handler = void (*)(const T*, const void*);
  template <typename T>
  using DecodeFunction = BoundedBuffer (*)(const char*, T*, BoundedBuffer,
                                           Handler<T>[], const void*);

  Portal() {}
  virtual ~Portal() {}

  template <typename T>
  SizeFunction<T> GetSizeFunction(const char* name) {
    return GetSizeFunction<T>(name, false);
  }
  template <typename T>
  EncodeFunction<T> GetEncodeFunction(const char* name) {
    return GetEncodeFunction<T>(name, false);
  }
  template <typename T>
  DecodeFunction<T> GetDecodeFunction(const char* name) {
    return GetDecodeFunction<T>(name, false);
  }

 protected:
  template <typename T>
  SizeFunction<T> GetSizeFunction(const char* name, bool internal) {
    return ResolveTarget<SizeFunction<T>>(name, internal);
  }
  template <typename T>
  EncodeFunction<T> GetEncodeFunction(const char* name, bool internal) {
    return ResolveTarget<EncodeFunction<T>>(name, internal);
  }
  template <typename T>
  DecodeFunction<T> GetDecodeFunction(const char* name, bool internal) {
    return ResolveTarget<DecodeFunction<T>>(name, internal);
  }

  template <typename T>
  T ResolveTarget(const char* name, bool internal) const {
    return reinterpret_cast<T>(ResolveTargetArtifact(name, internal));
  }

  virtual Artifact* ResolveTargetArtifact(const char* name,
                                          bool internal) const = 0;

  friend struct ProtoJitContext;
};

}  // namespace pj
