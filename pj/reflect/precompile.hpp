#include <cstdint>

#include "pj/protojit.hpp"
#include "pj/reflect/versions.hpp"

#define GET_PROTO_SIZE(V) V##_getProtoSize
#define ENCODE_PROTO(V) V##_encodeProto
#define DECODE_PROTO(V) V##_decodeProto

namespace pj {
namespace reflect {
namespace precomp {

#define EXTERN_GET_PROTO_SIZE(V) pj_reflect_##V##_getProtoSize
#define EXTERN_ENCODE_PROTO(V) pj_reflect_##V##_encodeProto
#define EXTERN_DECODE_PROTO(V) pj_reflect_##V##_decodeProto

#define EXTERN_DECLARATIONS(V, _)                                     \
  extern "C" size_t EXTERN_GET_PROTO_SIZE(V)(const Protocol*);        \
  extern "C" void EXTERN_ENCODE_PROTO(V)(const Protocol*, char* buf); \
  extern "C" BoundedBuffer EXTERN_DECODE_PROTO(V)(                    \
      const char* msg, Protocol* result, BoundedBuffer buffer,        \
      DecodeHandler<Protocol, void> handlers[], void* state);

FOR_EACH_COMPATIBLE_VERSION(EXTERN_DECLARATIONS)

#undef EXTERN_DECLARATIONS
}  // namespace precomp

#define DECLARATIONS(V, _)                                               \
  size_t GET_PROTO_SIZE(V)(const Protocol* proto) {                      \
    return precomp::EXTERN_GET_PROTO_SIZE(V)(proto);                     \
  }                                                                      \
  void ENCODE_PROTO(V)(const Protocol* proto, char* buf) {               \
    precomp::EXTERN_ENCODE_PROTO(V)(proto, buf);                         \
  }                                                                      \
  BoundedBuffer DECODE_PROTO(V)(const char* msg, Protocol* result,       \
                                BoundedBuffer buffer) {                  \
    return precomp::EXTERN_DECODE_PROTO(V)(msg, result, buffer, nullptr, \
                                           nullptr);                     \
  }

FOR_EACH_COMPATIBLE_VERSION(DECLARATIONS)

#undef DECLARATIONS

#undef EXTERN_GET_PROTO_SIZE
#undef EXTERN_ENCODE_PROTO
#undef EXTERN_DECODE_PROTO

}  // namespace reflect
}  // namespace pj
