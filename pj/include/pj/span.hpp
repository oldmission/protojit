#include <cstddef>
#include <stdexcept>

#include <pj/runtime.h>
#include <pj/arch_base.hpp>
#include <pj/traits.hpp>

namespace pj {

inline constexpr size_t unbounded_length = -1;  // NOLINT

template <typename wrapped, size_t max_length = unbounded_length,
          size_t min_size_hint = 0>
class span {  // NOLINT
 public:
  using T = typename wrapped_type<wrapped>::type;

  template <typename CharT, typename CharTraits,
            typename = std::enable_if_t<std::is_same_v<T, CharT>>>
  span(std::basic_string_view<CharT, CharTraits> view)
      : ptr_(view.data()), size_(view.size()) {
    validate<unbounded_length>();
  }

  template <typename Char,
            typename = std::enable_if_t<std::is_same_v<T, Char> &&
                                        std::is_same_v<Char, char>>>
  span(const Char* data) : ptr_(data), size_(strlen(data)) {
    validate<unbounded_length>();
  }

  span(const T* data, size_t size) : ptr_(data), size_(size) {
    validate<unbounded_length>();
  }

  span() : ptr_(nullptr), size_(0) {}

  template <size_t L, size_t H>
  span(const span<wrapped, L, H>& o) : ptr_(o.ptr_), size_(o.size_) {
    validate<L>();
  }

  template <size_t L, size_t H>
  span& operator=(const span<wrapped, L, H>& o) {
    ptr_ = o.ptr_, size_ = o.size;
    validate<L>();
    return *this;
  }

  const T& operator[](uintptr_t i) const { return ptr_[i]; }

  template <typename U>
  bool operator==(const U& o) const {
    return std::equal(begin(), end(), o.begin(), o.end());
  }

  size_t size() const { return size_; }
  const T* begin() const { return ptr_; }
  const T* end() const { return ptr_ + size_; }

 private:
  template <size_t other_max>
  void validate() {
    if constexpr (other_max > max_length) {
      if (max_length != unbounded_length && size_ > max_length) {
        throw std::logic_error("bad span");
      }
    }
  }

  template <typename Wrapped, size_t L, size_t H>
  friend class span;

  const T* ptr_;
  size_t size_;

  template <typename U>
  friend struct gen::BuildPJType;
};

namespace gen {
template <typename Elem, size_t MaxLength, size_t MinLength>
struct BuildPJType<::pj::span<Elem, MaxLength, MinLength>> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    using AV = ::pj::span<Elem, MaxLength>;
    return PJCreateVectorType(
        /*context=*/ctx,
        /*elem=*/BuildPJType<Elem>::build(ctx, domain),
        /*min_length=*/0,
        /*max_length=*/MaxLength,
        /*wire_min_length=*/MinLength,
        /*ppl_count=*/0,
        /*length_offset=*/offsetof(AV, size_) << 3,
        /*length_size=*/sizeof(AV::size_) << 3,
        /*ref_offset=*/offsetof(AV, ptr_) << 3,
        /*ref_size=*/sizeof(AV::ptr_) << 3,
        /*reference_mode=*/PJ_REFERENCE_MODE_POINTER,
        /*inline_payload_offset=*/kNone,
        /*inline_payload_size=*/kNone,
        /*partial_payload_offset=*/kNone,
        /*partial_payload_size=*/kNone,
        /*size=*/sizeof(AV) << 3,
        /*alignment=*/alignof(AV) << 3,
        /*outlined_payload_alignment=*/
        alignof(typename ::pj::wrapped_type<Elem>::type)  //
    );
  }
};
}  // namespace gen

}  // namespace pj
