#include <cstdint>

#include "pj/protojit.hpp"
#include "pj/schema/versions.hpp"

#define GET_PROTO_SIZE(V) V##_getProtoSize
#define ENCODE_PROTO(V) V##_encodeProto
#define DECODE_PROTO(V) V##_decodeProto

namespace pj {
namespace schema {
namespace precomp {

#define EXTERN_GET_PROTO_SIZE(V) pj_schema_##V##_getProtoSize
#define EXTERN_ENCODE_PROTO(V) pj_schema_##V##_encodeProto
#define EXTERN_DECODE_PROTO(V) pj_schema_##V##_decodeProto

#define EXTERN_DECLARATIONS(V, _)                                              \
  extern "C" size_t EXTERN_GET_PROTO_SIZE(V)(const reflect::Protocol*);        \
  extern "C" void EXTERN_ENCODE_PROTO(V)(const reflect::Protocol*, char* buf); \
  extern "C" BoundedBuffer EXTERN_DECODE_PROTO(V)(                             \
      const char* msg, reflect::Protocol* result, BoundedBuffer buffer,        \
      DecodeHandler<reflect::Protocol, void> handlers[], void* state);

FOR_EACH_COMPATIBLE_VERSION(EXTERN_DECLARATIONS)

#undef EXTERN_DECLARATIONS
}  // namespace precomp

#define DECLARATIONS(V, _)                                                  \
  size_t GET_PROTO_SIZE(V)(const reflect::Protocol* proto) {                \
    return precomp::EXTERN_GET_PROTO_SIZE(V)(proto);                        \
  }                                                                         \
  void ENCODE_PROTO(V)(const reflect::Protocol* proto, char* buf) {         \
    precomp::EXTERN_ENCODE_PROTO(V)(proto, buf);                            \
  }                                                                         \
  BoundedBuffer DECODE_PROTO(V)(const char* msg, reflect::Protocol* result, \
                                BoundedBuffer buffer) {                     \
    return precomp::EXTERN_DECODE_PROTO(V)(msg, result, buffer, nullptr,    \
                                           nullptr);                        \
  }

FOR_EACH_COMPATIBLE_VERSION(DECLARATIONS)

#undef DECLARATIONS

// Leave EXTERN_GET_PROTO_SIZE, EXTERN_ENCODE_PROTO, and EXTERN_DECODE_PROTO
// defined so that weak definitions can be generated for them in the
// non-precompiled version used for bootstrapping.

}  // namespace schema
}  // namespace pj
