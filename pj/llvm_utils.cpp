#include <llvm/IR/IntrinsicInst.h>

#include "llvm_utils.hpp"

namespace llvm_utils {
using namespace llvm;

std::optional<uint64_t> getGEPOffset(const DataLayout& layout,
                                     GetElementPtrInst* gep) {
  APInt gep_offset{layout.getIndexSizeInBits(gep->getPointerAddressSpace()), 0};
  if (!gep->accumulateConstantOffset(layout, gep_offset) ||
      gep_offset.isNegative()) {
    return {};
  }
  return gep_offset.getLimitedValue();
}

auto getRegionDerivation(const DataLayout& layout, Value* mem) {
  size_t offset = 0;
  SmallVector<std::pair<Value*, size_t>, 6> nodes{{mem, offset}};
  while (true) {
    if (auto* bitcast = dyn_cast<BitCastInst>(mem)) {
      mem = bitcast->getOperand(0);
      nodes.emplace_back(mem, offset);
    } else if (auto* gep = dyn_cast<GetElementPtrInst>(mem)) {
      if (auto gep_offset = getGEPOffset(layout, gep)) {
        mem = gep->getPointerOperand();
        offset += gep_offset.value();
        nodes.emplace_back(mem, offset);
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return nodes;
}

std::pair<Value*, size_t> getRegionRoot(const DataLayout& layout, Value* mem) {
  return getRegionDerivation(layout, mem).back();
}

void getDerivedLocations(
    const DataLayout& layout, Value* cur,
    llvm::SmallVector<std::pair<Value*, size_t>, 16>& derived,
    size_t offset = 0) {
  for (User* user : cur->users()) {
    if (auto* bitcast = dyn_cast<BitCastInst>(user)) {
      getDerivedLocations(layout, bitcast, derived, offset);
      derived.emplace_back(bitcast, offset);
    } else if (auto* gep = dyn_cast<GetElementPtrInst>(user)) {
      if (auto gep_offset = getGEPOffset(layout, gep)) {
        getDerivedLocations(layout, gep, derived, offset + gep_offset.value());
        derived.emplace_back(gep, offset + gep_offset.value());
      }
    }
  }
}

bool isMatchingFreeze(Value* val, size_t min_len) {
  if (auto* freeze = dyn_cast<IntrinsicInst>(val)) {
    if (freeze->getIntrinsicID() != Intrinsic::invariant_start) return false;
    auto freeze_len =
        cast<ConstantInt>(freeze->getOperand(0))->getLimitedValue();
    if (freeze_len >= min_len) {
      return true;
    }
  }
  return false;
}

bool isSafeToRead(const DataLayout& layout, Value* src, size_t offset,
                  size_t len) {
  // Search for an invariant.start intrinsic, generated from a ProjectOp with
  // frozen set to true, containing the region [src, src + len).
  for (auto [mem, start_offset] : getRegionDerivation(layout, src)) {
    auto total_offset = offset + start_offset;

    for (User* user : mem->users()) {
      if (isMatchingFreeze(user, total_offset + len)) {
        return true;
      }
    }

    // Also check the locations derived from mem for invariant.start intrinsics,
    // because LLVM optimization may have changed GEPs around.
    llvm::SmallVector<std::pair<Value*, size_t>, 16> derived_locations;
    getDerivedLocations(layout, mem, derived_locations);

    for (auto [derived, derived_offset] : derived_locations) {
      if (derived_offset > total_offset) continue;

      for (User* derived_user : derived->users()) {
        if (isMatchingFreeze(derived_user,
                             total_offset - derived_offset + len)) {
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace llvm_utils
