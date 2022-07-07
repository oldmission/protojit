#pragma once

#include <llvm/ADT/Hashing.h>
#include <llvm/Support/raw_ostream.h>

#include <pj/arch_base.hpp>

namespace pj {

inline ::llvm::hash_code hash_value(const Width& w) {
  return llvm::hash_code(w.bits());
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Width& w) {
  os << w.bits();
  return os;
}

}  // namespace pj
