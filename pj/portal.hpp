#pragma once

#include "util.hpp"

namespace pj {

struct Artifact {};

class Portal {
 public:
  Portal() {}
  virtual ~Portal() {}

  template <typename T>
  T ResolveTarget(const char* name, bool internal = false) const {
    return reinterpret_cast<T>(ResolveTargetArtifact(name, internal));
  }

  virtual Artifact* ResolveTargetArtifact(const char* name,
                                          bool internal) const = 0;
};

}  // namespace pj
