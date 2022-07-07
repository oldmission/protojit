#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <pj/util.hpp>

namespace pj {

template <typename T>
struct ArrayRef : public llvm::ArrayRef<T> {
  using llvm::ArrayRef<T>::ArrayRef;

  explicit ArrayRef(llvm::ArrayRef<T> o) : llvm::ArrayRef<T>(o) {}

  // Delete this error-prone constructor...
  ArrayRef(const T&) = delete;

  // ... and replace it with something more sane.
  ArrayRef(const T* o) : llvm::ArrayRef<T>(o, 1) {}
};

template <typename T>
struct ArrayRefConverter {
  ArrayRefConverter() {}

  template <typename Array, typename Convert = pj::Identity>
  ArrayRefConverter(const Array& arr, uintptr_t size, Convert&& convert = {}) {
    storage_.reserve(size);
    for (uintptr_t i = 0; i < size; ++i) {
      storage_.push_back(convert(arr[i]));
    }
  }

  template <typename Array, typename Convert = pj::Identity>
  ArrayRefConverter(const Array& arr, Convert&& convert = {})
      : ArrayRefConverter(arr, arr.size(), std::forward<Convert>(convert)) {}

  ArrayRefConverter(const ArrayRefConverter& o) : storage_{o.storage_} {}

  std::vector<T>& storage() { return storage_; }

  ArrayRef<T> get() { return ArrayRef<T>{storage_.data(), storage_.size()}; }

 private:
  std::vector<T> storage_;
};

}  // namespace pj
