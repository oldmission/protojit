#include "protojit.hpp"
#include "context.hpp"
#include "plan.hpp"

#include <llvm/Support/Debug.h>
#include <llvm/Support/TargetSelect.h>
#include <string.h>

#include <sstream>

namespace pj {

PJContext* getContext() {
  return reinterpret_cast<PJContext*>(new ProtoJitContext());
}

void freeContext(PJContext* ctx) {
  delete reinterpret_cast<ProtoJitContext*>(ctx);
}

std::unique_ptr<Portal> compile(PJContext* ctx) {
  return reinterpret_cast<ProtoJitContext*>(ctx)->compile();
}

}  // namespace pj
