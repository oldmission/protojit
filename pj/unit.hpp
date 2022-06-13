#include "runtime.h"
#include "traits.hpp"

namespace pj {

struct Unit {};

namespace gen {

template <>
struct BuildPJType<::pj::Unit> {
  static const auto* build(PJContext* ctx, const PJDomain* domain) {
    return PJCreateUnitType(ctx);
  }
};

}  // namespace gen
}  // namespace pj
