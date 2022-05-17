#include <map>

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Pass.h>
#include <llvm/Support/X86TargetParser.h>
#include <llvm/Target/TargetMachine.h>
#include "llvm/Analysis/AliasSetTracker.h"

#include "defer.hpp"
#include "passes.hpp"
#include "util.hpp"

namespace llvm {
void initializeCopyCoalescingPass(PassRegistry&);
}  // namespace llvm

using namespace llvm;

namespace {

struct CopyTargetInfo {
  explicit CopyTargetInfo(const llvm::TargetMachine& target) {
    if (target.getTargetTriple().isX86()) {
      min_page_size = 0x1000;
    }

    if (!target.getTargetTriple().isX86()) return;

    auto* subtarget = target.getMCSubtargetInfo();
    if (subtarget->hasFeature(X86::FEATURE_AVX512F)) {
      max_promotion_size = 64;
    } else if (subtarget->hasFeature(X86::FEATURE_AVX2)) {
      max_promotion_size = 32;
    } else if (subtarget->hasFeature(X86::FEATURE_AVX)) {
      max_promotion_size = 16;
    } else if (target.getTargetTriple().getArch() ==
               llvm::Triple::ArchType::x86_64) {
      max_promotion_size = 8;
    } else {
      max_promotion_size = 4;
    }
  }

  size_t max_promotion_size = 0;
  size_t min_page_size = 0;
};

struct Copy {
  static std::pair<llvm::Value*, size_t> getRegionRoot(const DataLayout& layout,
                                                       llvm::Value* mem) {
    size_t offset = 0;
    while (true) {
      if (auto* bitcast = dyn_cast<llvm::BitCastInst>(mem)) {
        mem = bitcast->getOperand(0);
      } else if (auto* gep = dyn_cast<llvm::GetElementPtrInst>(mem)) {
        APInt gep_offset{
            layout.getIndexSizeInBits(gep->getPointerAddressSpace()), 0};
        if (!gep->accumulateConstantOffset(layout, gep_offset) ||
            gep_offset.isNegative()) {
          break;
        }
        mem = gep->getPointerOperand();
        offset += gep_offset.getSExtValue();
      } else {
        break;
      }
    }
    return {mem, offset};
  }

  static std::optional<std::pair<BasicBlock::iterator, Copy>> match(
      AliasSetTracker& ast, const DataLayout& layout, llvm::Instruction* inst) {
    if (auto match = matchLoadAndStore(ast, layout, inst)) {
      return match;
    }
    if (auto match = matchPoison(ast, layout, inst)) {
      return match;
    }
    if (auto match = matchMemCpy(ast, layout, inst)) {
      return match;
    }
    return {};
  }

  static std::optional<std::pair<BasicBlock::iterator, Copy>> matchPoison(
      AliasSetTracker& ast, const DataLayout& layout, llvm::Instruction* inst) {
    IntrinsicInst* poison = dyn_cast<IntrinsicInst>(inst);
    if (!poison) return {};
    if (poison->getIntrinsicID() != Intrinsic::lifetime_end) return {};

    auto* len = dyn_cast<ConstantInt>(poison->getOperand(0));
    if (!len) return {};

    auto [store_root, store_base] =
        getRegionRoot(layout, poison->getOperand(1));

    ast.deleteValue(inst);
    auto it = poison->eraseFromParent();

    auto copy = Copy{
        .src_def = nullptr,
        .dst_def = store_root,
        .src_begin = 0,
        .dst_begin = store_base,
        .len = len->getZExtValue(),
    };
    return {{it, copy}};
  }

  static std::optional<std::pair<BasicBlock::iterator, Copy>> matchMemCpy(
      AliasSetTracker& ast, const DataLayout& layout, llvm::Instruction* inst) {
    MemCpyInst* memcpy = dyn_cast<MemCpyInst>(inst);
    if (!memcpy) return {};

    auto* len = dyn_cast<ConstantInt>(memcpy->getLength());
    if (!len) return {};

    auto [load_root, load_base] = getRegionRoot(layout, memcpy->getSource());
    auto [store_root, store_base] = getRegionRoot(layout, memcpy->getDest());

    ast.deleteValue(inst);
    auto it = memcpy->eraseFromParent();

    const auto copy = Copy{
        .src_def = load_root,
        .dst_def = store_root,
        .src_begin = load_base,
        .dst_begin = store_base,
        .len = len->getZExtValue(),
    };

    return {{it, copy}};
  }

  static std::optional<std::pair<BasicBlock::iterator, Copy>> matchLoadAndStore(
      AliasSetTracker& ast, const DataLayout& layout, llvm::Instruction* inst) {
    // Three conditions must hold for a match:
    //
    // 1. inst points to a load instruction with a single use
    // 2. the single use of inst is a store
    // 3. the source and destination memory regions cannot alias

    auto* load = dyn_cast<llvm::LoadInst>(inst);
    if (!load) return {};
    if (!load->hasNUses(1)) return {};

    auto* store = dyn_cast<llvm::StoreInst>(load->user_back());
    if (!store || store->getValueOperand() != load) return {};

    auto load_size = layout.getTypeStoreSize(load->getType());
    if (load_size.isScalable()) return {};

    auto width = load_size.getFixedSize();

    auto [load_root, load_base] =
        getRegionRoot(layout, load->getPointerOperand());
    auto [store_root, store_base] =
        getRegionRoot(layout, store->getPointerOperand());

    MemoryLocation load_region{load_root, load_base + width};
    MemoryLocation store_region{store_root, store_base + width};

    if (!ast.getAliasAnalysis().isNoAlias(load_region, store_region)) {
      return {};
    }

    ast.deleteValue(store);
    ast.deleteValue(load);
    auto it = store->eraseFromParent();
    load->eraseFromParent();

    const auto copy = Copy{
        .src_def = load_root,
        .dst_def = store_root,
        .src_begin = load_base,
        .dst_begin = store_base,
        .len = width,
    };

    return {{it, copy}};
  }

  llvm::Value* src_def;
  llvm::Value* dst_def;

  size_t src_begin;
  size_t dst_begin;

  size_t len;
};

struct CopySet {
  bool empty() const { return pieces.empty(); }

  bool canReorderWith(AAResults& aliasing,
                      const llvm::Instruction* inst) const {
    if (src_def == nullptr && dst_def == nullptr) {
      return true;
    }

    if (inst->isTerminator()) {
      return false;
    }

    assert(dst_def != nullptr);
    const MemoryLocation src_loc{src_def, load_end};
    const MemoryLocation dst_loc{dst_def, store_end};
    return (src_def == nullptr ||
            !isModSet(aliasing.getModRefInfo(inst, {src_loc}))) &&
           !isRefSet(aliasing.getModRefInfo(inst, {dst_loc}));
  }

  bool mergeWith(const Copy& copy) {
    if (dst_def == nullptr) {
      src_def = copy.src_def;
      dst_def = copy.dst_def;
    }

    if ((src_def != nullptr && copy.src_def != nullptr &&
         src_def != copy.src_def) ||
        dst_def != copy.dst_def) {
      return false;
    }

    if (src_def == nullptr) {
      src_def = copy.src_def;
    }

    // Check whether the store region overlaps with any store region in this
    // copy set.
    if (store_starts.size() > 0) {
      auto following = store_starts.lower_bound(copy.dst_begin);

      if (following != store_starts.end()) {
        auto [following_start, _] = *following;
        // Make sure this store doesn't overwrite the next one.
        if (copy.dst_begin + copy.len > following_start) {
          return false;
        }
      }

      Piece* preceeding = nullptr;
      if (following == store_starts.end()) {
        preceeding = store_starts.rbegin()->second;
      } else if (following != store_starts.begin()) {
        preceeding = (--following)->second;
      }

      if (preceeding) {
        auto preceeding_end = preceeding->store_start + preceeding->len;
        // Make sure the prior store doesn't overwrite this one.
        if (preceeding_end > copy.dst_begin) {
          return false;
        }
      }
    }

    pieces.push_back(Piece{
        .is_poison = copy.src_def == nullptr,
        .load_start = copy.src_begin,
        .store_start = copy.dst_begin,
        .len = copy.len,
    });

    store_starts.emplace(copy.dst_begin, &pieces.back());

    if (copy.src_def != nullptr) {
      load_starts.emplace(copy.src_begin, &pieces.back());
      load_end = std::max(load_end, copy.src_begin + copy.len);
    }
    store_end = std::max(store_end, copy.dst_begin + copy.len);

    return true;
  }

  void generate(const CopyTargetInfo& target, TargetLibraryInfo& tli,
                AliasSetTracker& ast, BasicBlock::iterator pos) const {
    if (pieces.size() == 0) {
      return;
    }

#if 0
    llvm::errs() << "---\n";
    for (auto& [_, piece] : store_starts) {
      if (piece->is_poison) {
        llvm::errs() << "poison dst " << piece->store_start << " -> "
                     << piece->store_start + piece->len << "\n";
      } else {
        llvm::errs() << "src " << piece->load_start << " -> "
                     << piece->load_start + piece->len << "  ==>  "
                     << piece->store_start << " -> "
                     << piece->store_start + piece->len << "\n";
      }
    }
    llvm::errs().flush();
#endif

    llvm::SmallVector<Piece*, 4> pieces;
    for (auto& [_, piece] : store_starts) pieces.push_back(piece);

    auto* it = pieces.begin();
    auto* piece = *it++;

    for (; it != pieces.end(); ++it) {
      // Combine this piece into the prior one if they are exactly adjacent.
      if (piece->is_poison == (*it)->is_poison &&
          piece->load_start + piece->len == (*it)->load_start &&
          piece->store_start + piece->len == (*it)->store_start) {
        piece->len += (*it)->len;
        continue;
      }

      auto try_extend_piece = [&]() {
        if (!llvm::isPowerOf2_64(piece->len) &&
            piece->len < target.max_promotion_size) {
          // If we can't combine them, try to extend the size of this piece to a
          // power of two. This requires:
          //
          // (a) we can ensure the trailing memory on the store side will be
          //     overwritten next.
          // (b) we can ensure the trailing memory on the load side is valid to
          //     read from, even if that read would entail reading arbitrary
          //     data.
          auto new_len = llvm::NextPowerOf2(piece->len);
          auto poison_len = new_len - piece->len;

          // (a)
          auto end = piece->store_start + piece->len;
          for (auto* j = it; j != pieces.end() && poison_len > 0; ++j) {
            if ((*j)->store_start > end) {
              break;
            }
            end += (*j)->len;
            poison_len -= std::min(poison_len, (*j)->len);
          }

          if (poison_len > 0) {
            return;
          }

          auto load_end = new_len + piece->load_start;
          auto pos = piece->load_start + piece->len;

          // (b)
          for (auto l = load_starts.lower_bound(piece->load_start);
               l != load_starts.end() && pos < load_end; ++l) {
            auto [_, next] = *l;
            if (next->load_start >= pos + target.min_page_size) break;
            pos = next->load_start + next->len;
          }

          if (pos < load_end) {
            return;
          }

          piece->len = new_len;
        }
      };

      try_extend_piece();

      piece->generate(ast, pos, src_def, dst_def);
      piece = *it;
    }

    piece->generate(ast, pos, src_def, dst_def);
  }

  llvm::Value* src_def = nullptr;
  llvm::Value* dst_def = nullptr;

  struct Piece {
    bool is_poison;
    size_t load_start;
    size_t store_start;
    size_t len;

    void generate(AliasSetTracker& ast, BasicBlock::iterator pos,
                  Value* src_def, Value* dst_def) {
      if (is_poison) return;
      IRBuilder<> builder(pos->getParent(), pos);
      auto* int8ptr = builder.getInt8PtrTy(0);
      auto* src_ptr = builder.CreateBitCast(src_def, int8ptr);
      auto* dst_ptr = builder.CreateBitCast(dst_def, int8ptr);
      auto* src_offset =
          builder.CreateConstGEP1_64(builder.getInt8Ty(), src_ptr, load_start);
      auto* dst_offset =
          builder.CreateConstGEP1_64(builder.getInt8Ty(), dst_ptr, store_start);
      ast.add(builder.CreateMemCpy(dst_offset, {}, src_offset, {}, len));
    };
  };

  std::list<Piece> pieces;
  std::map<size_t, Piece*> store_starts;
  std::map<size_t, Piece*> load_starts;

  size_t store_end = 0;
  size_t load_end = 0;
};

class CopyCoalescing : public FunctionPass {
 public:
  static char ID;
  CopyCoalescing() : FunctionPass(ID) {
    initializeCopyCoalescingPass(*PassRegistry::getPassRegistry());
  }

  CopyCoalescing(const TargetMachine& target)
      : FunctionPass(ID), target_(&target) {
    initializeCopyCoalescingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function& func) override {
    auto& tli = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(func);
    if (!tli.has(LibFunc_memcpy)) return false;

    auto& alias = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto ast = AliasSetTracker{alias};

    for (auto& block : func) {
      ast.add(block);
    }

    CopyTargetInfo cti{*target_};
    for (auto& block : func) {
      runOnBlock(cti, tli, ast, block);
    }

    return true;
  }

  void getAnalysisUsage(AnalysisUsage& usage) const override {
    usage.addRequired<AAResultsWrapperPass>();
    usage.addRequired<TargetLibraryInfoWrapperPass>();
  }

  void runOnBlock(const CopyTargetInfo& cti, TargetLibraryInfo& tli,
                  AliasSetTracker& ast, BasicBlock& bb) {
    auto it = bb.begin();
    CopySet set;
    while (it != bb.end()) {
      auto match = Copy::match(ast, target_->createDataLayout(), &*it);

      if (match.has_value()) {
        it = match->first;
        const auto& copy = match->second;

        if (set.mergeWith(copy)) {
          continue;
        }

        set.generate(cti, tli, ast, it);
        set = {};

        auto single = set.mergeWith(copy);
        ASSERT(single);
        continue;
      }

      DEFER(++it);

      if (!set.canReorderWith(ast.getAliasAnalysis(), &*it)) {
        set.generate(cti, tli, ast, it);
        set = {};
      }
    }
    assert(set.empty());
  }

 private:
  const TargetMachine* target_ = nullptr;
};
}  // namespace

char CopyCoalescing::ID = 0;
INITIALIZE_PASS(CopyCoalescing, "copy-coalescing", "Coalesce memory copies",
                false, false)

FunctionPass* llvm::createCopyCoalescingPass(const TargetMachine& target) {
  return new CopyCoalescing(target);
}
