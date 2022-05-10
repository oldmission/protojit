#pragma once

#include "util.hpp"

#include <llvm/ADT/ArrayRef.h>

namespace pj {

template <typename T>
struct Span : public llvm::ArrayRef<T> {
  using llvm::ArrayRef<T>::ArrayRef;

  explicit Span(llvm::ArrayRef<T> o) : llvm::ArrayRef<T>(o) {}

  // Delete this error-prone constructor...
  Span(const T&) = delete;

  // ... and replace it with something more sane.
  Span(const T* o) : llvm::ArrayRef<T>(o, 1) {}
};

template <typename T>
struct SpanConverter {
  SpanConverter() {}

  template <typename Array, typename Convert = pj::Identity>
  SpanConverter(const Array& arr, uintptr_t size, Convert&& convert = {}) {
    storage_.reserve(size);
    for (uintptr_t i = 0; i < size; ++i) {
      storage_.push_back(convert(arr[i]));
    }
  }

  template <typename Array, typename Convert = pj::Identity>
  SpanConverter(const Array& arr, Convert&& convert = {})
      : SpanConverter(arr, arr.size(), std::forward<Convert>(convert)) {}

  SpanConverter(const SpanConverter& o) : storage_{o.storage_} {}

  std::vector<T>& storage() { return storage_; }

  Span<T> get() { return Span<T>{&storage_[0], storage_.size()}; }

 private:
  std::vector<T> storage_;
};

}  // namespace pj
