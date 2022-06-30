#pragma once

#include <cstdint>

#include "arch.hpp"
#include "runtime.h"

namespace pj {

namespace reflect {

struct Type;

struct ReflectionTypeVector {
  ReflectionTypeVector() : offset_(0), size_(0) {}

  ReflectionTypeVector(const Type* data, size_t size) : size_(size) {
    offset_ =
        reinterpret_cast<intptr_t>(data) - reinterpret_cast<intptr_t>(this);
  }

  ~ReflectionTypeVector() {}

  const Type* base() const {
    return reinterpret_cast<const Type*>(  //
        reinterpret_cast<intptr_t>(this) + offset_);
  }

  size_t size() const { return size_; }

  ReflectionTypeVector(const ReflectionTypeVector& other) : size_(other.size_) {
    offset_ = other.offset_ + reinterpret_cast<intptr_t>(&other) -
              reinterpret_cast<intptr_t>(this);
  }

  ReflectionTypeVector& operator=(const ReflectionTypeVector& other) = delete;
  ReflectionTypeVector(ReflectionTypeVector&& other) = delete;
  ReflectionTypeVector& operator=(ReflectionTypeVector&& other) = delete;

 private:
  friend gen::BuildPJType<reflect::ReflectionTypeVector>;

  intptr_t offset_;
  size_t size_;
};

struct Protocol {
  int32_t pj_version;
  int32_t head;
  Width buffer_offset;
  ReflectionTypeVector types;
};

}  // namespace reflect

namespace gen {

template <>
struct BuildPJType<reflect::ReflectionTypeVector> {
  static const PJVectorType* build(PJContext* ctx, const PJDomain* domain);
};

}  // namespace gen

}  // namespace pj
