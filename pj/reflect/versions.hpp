#include "pj/reflect/reflect.pj.v0_1.hpp"

#include <cstdint>

#define CURRENT_VERSION 0x0000'0001'0000'0000ul

// Defines the version name, and a 64-bit version number for that version.
#define FOR_EACH_COMPATIBLE_VERSION(V) V(v0_1, CURRENT_VERSION)
