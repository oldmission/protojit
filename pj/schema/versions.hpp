#include "pj/schema/reflect.pj.v0_1.hpp"
#include "pj/schema/reflect.pj.v0_2.hpp"

#include <cstdint>

#define CURRENT_VERSION 0x0000'0002'0000'0000ul

// Defines the version name, and a 64-bit version number for that version.
#define FOR_EACH_COMPATIBLE_VERSION(V) \
  V(v0_1, 0x0000'0001'0000'0000ul)     \
  V(v0_2, CURRENT_VERSION)
