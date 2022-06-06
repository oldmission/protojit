#include "pj/ir.hpp"

namespace pj {
namespace reflect {

struct Protocol;
Protocol reflect(llvm::BumpPtrAllocator& alloc, types::ProtocolType proto);
types::ValueType unreflect(const Protocol& type, mlir::MLIRContext& ctx);
types::ValueType reflectableTypeFor(types::ValueType);

}  // namespace reflect
}  // namespace pj
