#pragma once

#include <cassert>
#include <cinttypes>

#include <llvm/ADT/Hashing.h>
#include <llvm/Support/raw_ostream.h>

namespace pj {

constexpr inline intptr_t kByte = 8;
constexpr inline intptr_t kMaxCppIntSize = 8;
constexpr inline intptr_t kNone = -1;

enum class Arch {
  kX64,
};

struct ArchDetails {
  const Arch arch;
  const intptr_t word_size_bytes;

  static ArchDetails Host() {
    return ArchDetails{
        .arch = Arch::kX64,
        .word_size_bytes = 8,
    };
  }
};

struct Width;
inline Width Bits(intptr_t bits);
inline Width Bytes(intptr_t bytes);

struct Width {
  intptr_t bits() const { return bits_; }
  intptr_t bytes() const {
    if (bits_ == kNone) return bits_;
    assert(bits_ % kByte == 0);
    return bits_ / kByte;
  }

  bool operator<(const Width& other) const {
    assert(IsNotNone() && other.IsNotNone());
    return bits_ < other.bits_;
  }
  bool operator>(const Width& other) const {
    assert(IsNotNone() && other.IsNotNone());
    return bits_ > other.bits_;
  }

  bool operator==(const Width& other) const { return bits_ == other.bits_; }
  bool operator!=(const Width& other) const { return bits_ != other.bits_; }

  Width operator+(const Width& other) const {
    assert(IsNotNone() && other.IsNotNone());
    return Bits(bits() + other.bits());
  }

  Width operator-(const Width& other) const {
    assert(IsNotNone() && other.IsNotNone());
    return Bits(bits() - other.bits());
  }

  void operator+=(const Width& other) {
    assert(IsNotNone() && other.IsNotNone());
    bits_ += other.bits_;
  }
  void operator-=(const Width& other) {
    assert(IsNotNone() && other.IsNotNone());
    bits_ -= other.bits_;
  }

  intptr_t operator/(const Width& other) const {
    assert(bits_ % other.bits_ == 0);
    return bits_ / other.bits_;
  }

  Width operator*(intptr_t val) const { return Bits(bits_ * val); }

  static Width None() { return Width{kNone}; }

  bool IsNone() const { return bits_ == kNone; }
  bool IsNotNone() const { return bits_ != kNone; }
  bool IsBytes() const { return bits_ == kNone || bits_ % kByte == 0; }

  bool IsZero() const { return bits_ == 0; }
  bool IsPos() const { return bits_ > 0; }

  bool IsAlignedTo(const Width& other) const {
    return bits_ % other.bits_ == 0;
  }

 private:
  friend Width Bits(intptr_t bits);
  friend Width Bytes(intptr_t bits);

  explicit Width(intptr_t bits) : bits_(bits) {}
  intptr_t bits_;
};

inline ::llvm::hash_code hash_value(const Width& w) {
  return llvm::hash_code(w.bits());
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Width& W) {
  os << W.bits();
  return os;
}

inline Width Bits(intptr_t bits) { return Width(bits); }
inline Width Bytes(intptr_t bytes) {
  return bytes == kNone ? Width(bytes) : Width(bytes * kByte);
}

}  // namespace pj
