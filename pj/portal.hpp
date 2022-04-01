#pragma once

#include "target.hpp"
#include "util.hpp"

namespace pj {

class PortalSpec : public Scoped {
 public:
  const std::vector<const Target*> targets;

  PortalSpec(std::vector<const Target*>&& targets) : targets(targets) {}
};

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
