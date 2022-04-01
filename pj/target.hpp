#pragma once

#include "concrete_types.hpp"
#include "tag.hpp"

namespace mlir {
class FuncOp;
class MLIRContext;
}  // namespace mlir

namespace pj {

class Protocol;
class Portal;
struct Artifact;

#define FOR_EACH_TARGET(V) \
  V(Size)                  \
  V(Encode)                \
  V(Decode)

#define DECLARE(name) class name##Target;
FOR_EACH_TARGET(DECLARE)
#undef DECLARE

class Target : public Scoped {
 public:
  Target(std::string&& name) : name(name) {}
  virtual ~Target();

  virtual mlir::FuncOp Compile(const ArchDetails& details, Scope* S,
                               mlir::MLIRContext* C) const = 0;

  intptr_t ByteAlignment(const ArchDetails& arch) const {
    return arch.word_size_bytes;
  }

  intptr_t ByteSize(const ArchDetails& arch) const {
    return arch.word_size_bytes;
  }

#define HANDLE_TARGET(name)                 \
  virtual bool Is##name() { return false; } \
  virtual name##Target* As##name() { throw std::logic_error("Target cast"); }

  FOR_EACH_TARGET(HANDLE_TARGET)
#undef HANDLE_TARGET

  std::string name;
};

#define MAKE_TARGET(name)                                    \
 public:                                                     \
  bool Is##name() override { return true; }                  \
  name##Target* As##name() override { return this; }         \
  mlir::FuncOp Compile(const ArchDetails& details, Scope* S, \
                       mlir::MLIRContext* C) const override;

class SizeTarget : public Target {
 public:
  const Path path;
  const CType* const mem;
  const Protocol* const proto;

  SizeTarget(std::string&& name, const Path& path, const CType* mem,
             const Protocol* proto)
      : Target(std::move(name)), path(path), mem(mem), proto(proto) {}

  MAKE_TARGET(Size);
};

class EncodeTarget : public Target {
 public:
  const Path path;
  const CType* const mem;
  const Protocol* const proto;

  EncodeTarget(std::string&& name, const Path& path, const CType* mem,
               const Protocol* proto)
      : Target(std::move(name)), path(path), mem(mem), proto(proto) {}

  MAKE_TARGET(Encode);
};

class DecodeTarget : public Target {
 public:
  const Protocol* const proto;
  const CType* const mem;
  Path dispatch_path;
  const std::map<std::string, const void*> branches;

  DecodeTarget(std::string&& name, const Protocol* proto, const CType* mem,
               Path&& dispatch_path, decltype(branches)&& branches)
      : Target(std::move(name)),
        proto(proto),
        mem(mem),
        dispatch_path(std::move(dispatch_path)),
        branches(branches) {}

  MAKE_TARGET(Decode);
};

}  // namespace pj
