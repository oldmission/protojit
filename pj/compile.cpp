#include "context.hpp"

namespace pj {

std::unique_ptr<Portal> Compile(Scope* scope, const PortalSpec* spec) {
  const ArchDetails arch = ArchDetails::Host();

  ProtoJitContext ctx;

  for (auto* target : spec->targets) {
    ctx.module_->push_back(target->Compile(arch, scope, &ctx.ctx_));
  }

  return ctx.compile();
}

#undef DEBUG_TYPE

}  // namespace pj
