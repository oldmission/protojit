#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "protojit.hpp"

namespace pj {

struct Artifact {};

class Portal {
 public:
  Portal() {}
  virtual ~Portal() {}

  template <typename T>
  SizeFunction<T> GetSizeFunction(const char* name,
                                  bool internal = false) const {
    return ResolveTarget<SizeFunction<T>>(name, internal);
  }
  template <typename T>
  EncodeFunction<T> GetEncodeFunction(const char* name,
                                      bool internal = false) const {
    return ResolveTarget<EncodeFunction<T>>(name, internal);
  }
  template <typename T, typename BBuf = BoundedBuffer>
  DecodeFunction<T, BBuf> GetDecodeFunction(const char* name,
                                            bool internal = false) const {
    return ResolveTarget<DecodeFunction<T, BBuf>>(name, internal);
  }

  template <typename T>
  T ResolveTarget(const char* name, bool internal) const {
    return reinterpret_cast<T>(ResolveTargetArtifact(name, internal));
  }

  virtual Artifact* ResolveTargetArtifact(const char* name,
                                          bool internal) const = 0;
};

}  // namespace pj
