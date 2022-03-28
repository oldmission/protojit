#include "abstract_types.hpp"

namespace pj {
AType::~AType() {}

#define MAKE_DTOR(name) \
  A##name##Type::~A##name##Type() {}
FOR_EACH_TYPE(MAKE_DTOR)
#undef MAKE_DTOR

}  // namespace pj
