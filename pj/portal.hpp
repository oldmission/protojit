#pragma once

#include "util.hpp"

namespace pj {

struct Artifact {};

class Portal {
 public:
  Portal() {}
  virtual ~Portal() {}

  template <typename T>
  T ResolveTarget(const char* name) const {
    return reinterpret_cast<T>(ResolveTargetArtifact(name));
  }

  virtual Artifact* ResolveTargetArtifact(const char* name) const = 0;
};

}  // namespace pj
