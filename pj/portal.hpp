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
  SizeFunction<T> GetSizeFunction(const char* name) const {
    return ResolveTarget<SizeFunction<T>>(name);
  }
  template <typename T>
  EncodeFunction<T> GetEncodeFunction(const char* name) const {
    return ResolveTarget<EncodeFunction<T>>(name);
  }
  template <typename T, typename BBuf = BoundedBuffer>
  DecodeFunction<T, BBuf> GetDecodeFunction(const char* name) const {
    return ResolveTarget<DecodeFunction<T, BBuf>>(name);
  }

  template <typename T>
  T ResolveTarget(const char* name) const {
    return reinterpret_cast<T>(ResolveTargetArtifact(name));
  }

  virtual Artifact* ResolveTargetArtifact(const char* name) const = 0;
};

}  // namespace pj
