#pragma once

#include "concrete_types.hpp"
#include "tag.hpp"

namespace pj {

struct ProtoParams {
  Width max_size = Width::None();
};

class ProtoSpec : public Scoped {
 public:
  const Path tag;
  const CType* const head;
  const ProtoParams params;

  ProtoSpec(const Path& tag, const CType* head, const ProtoParams& params)
      : tag(tag), head(head), params(params) {}

  void Validate() const;
};

class Protocol : public Scoped {
 public:
  const CType* const head;
  const Path tag;
  const Width tag_size_;

  Protocol(const CType* head, Path&& tag, Width tag_size)
      : head(head), tag(tag), tag_size_(tag_size) {
    assert(head != nullptr);
  }

  void Validate() const;

  // For serialize
  Width SizeOf(const CType* from, const Path& path) const;

  // TODO: requires fixups, for deserialize
};

}  // namespace pj
