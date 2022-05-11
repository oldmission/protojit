#include "pj/ir.hpp"

namespace pj {
namespace reflect {

struct Proto;
Proto reflect(llvm::BumpPtrAllocator& alloc, types::ProtocolType proto);
types::ValueType reflectableTypeFor(types::ValueType);

}  // namespace reflect
}  // namespace pj
