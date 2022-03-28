#include "protojit.hpp"

#include <llvm/Support/Debug.h>
#include <llvm/Support/TargetSelect.h>
#include <string.h>

#include <functional>
#include <sstream>

namespace pj {

__attribute__((constructor)) //
void Init() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

#ifndef NDEBUG
  const char* env = getenv("PROTOJIT_INTERNAL_DEBUG_TYPES");
  if (env != nullptr && env[0] != '\0') {
    if (!strcmp(env, "ON") || !strcmp(env, "1")) {
      llvm::DebugFlag = 1;
    } else {
      std::istringstream str(env);
      std::string type;
      std::vector<std::string> types;
      std::vector<const char*> types_p;
      while (getline(str, type, ',')) {
        types.push_back(type);
        types_p.push_back(types.back().c_str());
      }

      if (types.size() > 0) {
        llvm::DebugFlag = 1;
        llvm::setCurrentDebugTypes(types_p.data(), types_p.size());
      }
    }
  }
#endif
}

const Protocol* Negotiate(Scope* scope, const ProtoSpec* recv,
                          const ProtoSpec* send) {
  throw IssueError(12);
}

void Validate(const Protocol* proto, const ProtoSpec* spec, Side side,
              NegotiateOptions opts) {
  throw IssueError(8);
}

const Protocol* PlanProtocol(Scope* scope, const ProtoSpec* spec) {
  spec->Validate();

  auto tag_type = spec->head->Resolve(spec->tag);
  const Width tag_size =
      tag_type != nullptr ? tag_type->AsVariant()->tag_size : Bits(0);

  auto head = spec->head->Plan(*scope, spec->params, spec->tag);
  return new (scope) Protocol(head, Path{spec->tag}, tag_size);
}

}  // namespace pj
