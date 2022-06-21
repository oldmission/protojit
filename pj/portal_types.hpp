#pragma once

namespace pj {

struct BoundedBuffer {
  char* ptr;
  uint64_t size;
};

static_assert(sizeof(BoundedBuffer) == 2 * sizeof(void*));
static_assert(sizeof(BoundedBuffer::ptr) == sizeof(void*));
static_assert(offsetof(BoundedBuffer, ptr) == 0);
static_assert(offsetof(BoundedBuffer, size) == sizeof(void*));

template <typename T>
using Handler = void (*)(const T*, const void*);
template <typename T>
using SizeFunction = uintptr_t (*)(const T*);
template <typename T>
using EncodeFunction = void (*)(const T*, char*);
template <typename T, typename BBuf>
using DecodeFunction = BBuf (*)(const char*, T*, BBuf, Handler<T>[],
                                const void*);

}  // namespace pj
