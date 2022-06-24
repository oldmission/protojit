#pragma once

#include <string>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/RuntimeDyld.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

#include "portal.hpp"

namespace pj {

class PortalImpl : public Portal {
 public:
  PortalImpl(std::unique_ptr<llvm::orc::LLJIT>&& jit) : jit_(std::move(jit)) {}
  ~PortalImpl() {}

  std::unique_ptr<llvm::orc::LLJIT> jit_;

  Artifact* ResolveTargetArtifact(const char* name) const override {
    auto result = jit_->lookup(name);
    if (!result) {
      return nullptr;
    } else {
      return reinterpret_cast<Artifact*>(result->getAddress());
    }
  }

  DISALLOW_COPY_AND_ASSIGN(PortalImpl);
};

}  // namespace pj
