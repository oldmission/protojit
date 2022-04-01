#pragma once

#include "arch.hpp"
#include "exceptions.hpp"

namespace pj {

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)

#define UNREACHABLE()                           \
  assert(false && "Unreachable!"); /* NOLINT */ \
  throw ::pj::InternalError("Unreachable!");

#ifndef NDEBUG
#define ASSERT(X) assert(X)
#else
#define ASSERT(X) if (false) { (void)(X); }
#endif

template <typename T>
inline T DivideUp(T x, T multiple);

template <typename T>
inline T RoundUp(T x, T multiple);

inline intptr_t DivideUp(Width x, Width y) {
  return DivideUp(x.bits(), y.bits());
}

inline intptr_t DivideDown(Width x, Width y) { return x.bits() / y.bits(); }

inline Width RoundUp(Width x, Width y) {
  return Bits(RoundUp(x.bits(), y.bits()));
}

template <typename T>
inline T DivideUp(T x, T multiple) {
  return (x + multiple - 1) / multiple;
}

template <typename T>
inline T RoundUp(T x, T multiple) {
  return DivideUp(x, multiple) * multiple;
}

// Stand-in for C++20 std::identity
struct Identity {
  template <typename T>
  constexpr T&& operator()(T&& t) const noexcept {
    return std::forward<T>(t);
  }
};

// Copied from compiler-rt

template <typename T>
static inline bool IsPowerOfTwo(T X) {
  return (X & (X - 1)) == 0;
}

static inline intptr_t GetMostSignificantSetBitIndex(intptr_t X) {
  // SAMIR_TODO: 64
  return 64 - 1U - static_cast<intptr_t>(__builtin_clzl(X));
}

inline intptr_t RoundUpToPowerOfTwo(intptr_t size) {
  if (IsPowerOfTwo(size)) {
    return size;
  }
  const intptr_t highest = GetMostSignificantSetBitIndex(size);
  return 1UL << (highest + 1);
}

}  // namespace pj

#ifndef NDEBUG
#define DEBUG_ONLY(X) X
#else
#define DEBUG_ONLY(X)
#endif
