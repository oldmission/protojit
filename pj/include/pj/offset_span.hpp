#pragma once

#include <cstddef>
#include <cstdint>

#include "runtime.hpp"
#include "traits.hpp"

namespace pj {
template <typename wrapped>
struct offset_span {  // NOLINT
  using T = typename pj::wrapped_type<wrapped>::type;

  offset_span() : offset_(0), size_(0) {}

  offset_span(const T* data, size_t size) : size_(size) {
    offset_ =
        reinterpret_cast<intptr_t>(data) - reinterpret_cast<intptr_t>(this);
  }

  const T* base() const {
    return reinterpret_cast<const T*>(reinterpret_cast<intptr_t>(this) +
                                      offset_);
  }

  size_t size() const { return size_; }

  offset_span(const offset_span& other) : size_(other.size_) {
    offset_ = other.base_char() - reinterpret_cast<const char*>(this);
  }

  offset_span& operator=(const offset_span& other) {
    offset_ = other.base_char() - reinterpret_cast<const char*>(this);
    size_ = other.size_;
    return *this;
  }

  offset_span(offset_span&& other) : size_(other.size_) {
    offset_ = other.base_char() - reinterpret_cast<const char*>(this);
  }

  offset_span& operator=(offset_span&& other) {
    offset_ = other.base_char() - reinterpret_cast<const char*>(this);
    size_ = other.size_;
    return *this;
  }

  const T* begin() const { return base(); }
  const T* end() const { return base() + size(); }

  const T& operator[](size_t i) const { return *(base() + i); }

 private:
  const char* base_char() const {
    return reinterpret_cast<const char*>(base());
  }

  friend gen::BuildPJType<offset_span>;

  intptr_t offset_;
  size_t size_;
};

namespace gen {

template <typename wrapped>
struct BuildPJType<offset_span<wrapped>> {
  static const PJVectorType* build(PJContext* ctx, const PJDomain* domain) {
    using Span = offset_span<wrapped>;
    // Offsets are interpreted as being an offset from the beginning byte of the
    // offset itself. Make sure that's the same as the beginning of the object.
    static_assert(offsetof(Span, offset_) == 0);
    using T = typename pj::wrapped_type<wrapped>::type;
    return PJCreateVectorType(  //
        ctx,
        /*elem=*/BuildPJType<wrapped>::build(ctx, domain),
        /*min_length=*/0,
        /*max_length=*/-1,
        /*wire_min_length=*/0,
        /*ppl_count=*/0,
        /*length_offset=*/offsetof(Span, size_) * 8,
        /*length_size=*/sizeof(Span::size_) * 8,
        /*ref_offset=*/offsetof(Span, offset_) * 8,
        /*ref_size=*/sizeof(Span::offset_) * 8,
        /*reference_mode=*/PJ_REFERENCE_MODE_OFFSET,
        /*inline_payload_offset=*/-1,
        /*inline_payload_size=*/-1,
        /*partial_payload_offset=*/-1,
        /*partial_payload_size=*/-1,
        /*size=*/sizeof(Span) * 8,
        /*alignment=*/alignof(Span) * 8,
        /*outlined_payload_alignment=*/alignof(T) * 8  //
    );
  }
};

}  // namespace gen
}  // namespace pj
