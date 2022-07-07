#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/RuntimeDyld.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

#include <pj/runtime.hpp>

namespace pj {

struct Artifact {};

class Portal {
 public:
  Portal(std::unique_ptr<llvm::orc::LLJIT>&& jit) : jit_(std::move(jit)) {}
  ~Portal() {}

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

  Artifact* ResolveTargetArtifact(const char* name) const {
    auto result = jit_->lookup(name);
    if (!result) {
      return nullptr;
    } else {
      return reinterpret_cast<Artifact*>(result->getAddress());
    }
  }

 private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
};

}  // namespace pj
