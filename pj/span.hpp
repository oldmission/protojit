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
  template <typename Array, typename Convert = pj::Identity>
  SpanConverter(const Array& arr, uintptr_t size, Convert&& convert = {}) {
    storage.reserve(size);
    for (uintptr_t i = 0; i < size; ++i) {
      storage.push_back(convert(arr[i]));
    }
  }

  template <typename Array, typename Convert = pj::Identity>
  SpanConverter(const Array& arr, Convert&& convert = {})
      : SpanConverter(arr, arr.size(), std::forward<Convert>(convert)) {}

  Span<T> get() { return Span<T>{&storage[0], storage.size()}; }

 private:
  std::vector<T> storage;
};

}  // namespace pj
