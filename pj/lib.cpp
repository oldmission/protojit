#include <llvm/Support/Debug.h>
#include <llvm/Support/TargetSelect.h>

#include <cstring>
#include <sstream>
#include <vector>

__attribute__((constructor)) //
void init() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

#ifndef NDEBUG
  const char* env = getenv("PROTOJIT_INTERNAL_DEBUG_TYPES");
  if (env != nullptr && env[0] != '\0') {
    if (!std::strcmp(env, "ON") || !std::strcmp(env, "1")) {
      llvm::DebugFlag = 1;
    } else {
      std::istringstream str(env);
      std::string type;
      std::vector<std::string> types;
      std::vector<const char*> types_p;
      while (getline(str, type, ',')) {
        types.push_back(type);
      }

      for (const std::string& type : types) {
        types_p.push_back(type.c_str());
      }

      if (types.size() > 0) {
        llvm::DebugFlag = 1;
        llvm::setCurrentDebugTypes(types_p.data(), types_p.size());
      }
    }
  }
#endif
}
