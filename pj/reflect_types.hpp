#pragma once

#include <cstdint>

#include "arch.hpp"
#include "protojit.hpp"
#include "runtime.h"

namespace pj {

namespace reflect {

struct Type;

struct StructField {
  int32_t type;
  offset_span<char, Integer<char, PJSign::PJ_SIGN_SIGNLESS>> name;
  Width offset;
};

struct Protocol {
  int32_t pj_version;
  int32_t head;
  Width buffer_offset;
  offset_span<Type> types;
};

}  // namespace reflect

}  // namespace pj
