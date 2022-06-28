#include <cstdio>

#include "pj/runtime.hpp"
#include "pj/schema/precompile.hpp"
#include "pj/util.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "No output filename given");
    return 1;
  }

  PJContext* ctx = PJGetContext();

  auto host_domain = PJGetHostDomain(ctx);
  auto host_type =
      pj::gen::BuildPJType<pj::reflect::Protocol>::build(ctx, host_domain);

#define ADD_FUNCTIONS(V, _)                                                 \
  {                                                                         \
    auto proto = pj::gen::BuildPJProtocol<::V::pj::reflect::Schema>::build( \
        ctx, PJGetWireDomain(ctx));                                         \
    PJAddSizeFunction(ctx, "pj_schema_" STR(V) "_getProtoSize", host_type,  \
                      proto, "", true);                                     \
    PJAddEncodeFunction(ctx, "pj_schema_" STR(V) "_encodeProto", host_type, \
                        proto, "");                                         \
    PJAddDecodeFunction(ctx, "pj_schema_" STR(V) "_decodeProto", proto,     \
                        host_type, 0, nullptr);                             \
  }

  FOR_EACH_COMPATIBLE_VERSION(ADD_FUNCTIONS)

#undef ADD_FUNCTIONS

  PJPrecompile(ctx, argv[1], true);
}
