#pragma once

#include <string>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/RuntimeDyld.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

#include "portal.hpp"

namespace pj {

static constexpr const char* kUserFunctionPrefix = "user_";

class PortalImpl : public Portal {
 public:
  PortalImpl(std::unique_ptr<llvm::orc::LLJIT>&& jit) : jit_(std::move(jit)) {}
  ~PortalImpl() {}

  std::unique_ptr<llvm::orc::LLJIT> jit_;

  Artifact* ResolveTargetArtifact(const char* name,
                                  bool internal = false) const override {
    auto full_name = (internal ? "" : kUserFunctionPrefix) + std::string{name};
    auto result = jit_->lookup(full_name);
    if (!result) {
      return nullptr;
    } else {
      return reinterpret_cast<Artifact*>(result->getAddress());
    }
  }

  DISALLOW_COPY_AND_ASSIGN(PortalImpl);
};

}  // namespace pj
