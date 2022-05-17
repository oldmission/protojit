#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>

#include <optional>
#include <tuple>

namespace llvm_utils {

// Finds the rootmost value leading to the provided pointer via bitcasts and
// GEPs along with the total offset from the root value to the pointer.
std::pair<llvm::Value*, size_t> getRegionRoot(const llvm::DataLayout& layout,
                                              llvm::Value* mem);

// Searches for an invariant.start intrinsic, generated from a ProjectOp with
// frozen set to true, containing the region [src, src + len).
bool isSafeToRead(const llvm::DataLayout& layout, llvm::Value* src,
                  size_t offset, size_t len);

}  // namespace llvm_utils
