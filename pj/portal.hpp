#pragma once

#include <tuple>

namespace pj {

struct ProtoJitContext;

struct Artifact {};

class Portal {
 public:
  struct BoundedBuffer {
    char* ptr;
    uint64_t size;
  };
  template <typename T>
  using SizeFunction = uintptr_t (*)(const T*);
  template <typename T>
  using EncodeFunction = void (*)(const T*, char*);
  template <typename T>
  using Handler = void (*)(const T*, const void*);
  template <typename T, typename BBuf>
  using DecodeFunction = BBuf (*)(const char*, T*, BBuf, Handler<T>[],
                                  const void*);

  Portal() {}
  virtual ~Portal() {}

  template <typename T>
  SizeFunction<T> GetSizeFunction(const char* name) const {
    return GetSizeFunction<T>(name, false);
  }
  template <typename T>
  EncodeFunction<T> GetEncodeFunction(const char* name) const {
    return GetEncodeFunction<T>(name, false);
  }
  template <typename T, typename BBuf = BoundedBuffer>
  DecodeFunction<T, BBuf> GetDecodeFunction(const char* name) const {
    return GetDecodeFunction<T, BBuf>(name, false);
  }

 protected:
  template <typename T>
  SizeFunction<T> GetSizeFunction(const char* name, bool internal) const {
    return ResolveTarget<SizeFunction<T>>(name, internal);
  }
  template <typename T>
  EncodeFunction<T> GetEncodeFunction(const char* name, bool internal) const {
    return ResolveTarget<EncodeFunction<T>>(name, internal);
  }
  template <typename T, typename BBuf = BoundedBuffer>
  DecodeFunction<T, BBuf> GetDecodeFunction(const char* name,
                                            bool internal) const {
    return ResolveTarget<DecodeFunction<T, BBuf>>(name, internal);
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
