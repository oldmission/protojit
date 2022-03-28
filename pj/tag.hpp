#pragma once

#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <map>

#include "abstract_types.hpp"
#include "concrete_types.hpp"
#include "scope.hpp"

namespace pj {

inline bool IsEmptyTag(PathPiece tag) { return tag.begin == tag.end; }

inline PathPiece Tail(PathPiece tag) {
  if (IsEmptyTag(tag)) {
    return tag;
  } else {
    return PathPiece{tag.begin + 1, tag.end};
  }
}

inline bool IsDotTag(PathPiece tag) {
  return tag.begin != tag.end && *tag.begin == ".";
}

inline PathPiece Narrow(PathPiece tag, std::string_view head) {
  return tag.begin != tag.end && *tag.begin == head ? Tail(tag) : PathPiece{};
}

}  // namespace pj
