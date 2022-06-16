#pragma once

#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "pj/portal.hpp"
#include "pj/runtime.hpp"

namespace pj {

template <size_t I>
static void handleDispatch(
    const void* received,
    std::vector<std::function<void(const void*)>>* state) {
  return (*state)[I](received);
}

template <bool VariantTest = false>
class PJGenericTest
    : public std::conditional_t<VariantTest,
                                ::testing::TestWithParam<std::pair<bool, bool>>,
                                ::testing::Test> {
 protected:
  void SetUp() override {
    ctx = std::make_unique<runtime::Context>();
    if constexpr (VariantTest) {
      std::tie(no_tag, no_src_path) = this->GetParam();
    } else {
      no_tag = false;
      no_src_path = false;
    }
  }

  ~PJGenericTest() { EXPECT_TRUE(waiting_matches.empty()); }

  template <size_t I, typename T>
  void onMatch(const std::string& term,
               typename std::function<void(const T&)> callback) {
    waiting_matches.insert(term);
    _on<I, T>(term, [this, term, callback = std::move(callback)](const T& t) {
      waiting_matches.erase(term);
      callback(t);
    });
  }

  template <size_t I, typename T>
  void onNoMatch(const std::string& term,
                 typename std::function<void(const T&)> callback) {
    _on<I, T>(term, [this, term, callback = std::move(callback)](const T& t) {
      waiting_matches.insert(term);
      callback(t);
    });
  }

  template <size_t I, typename T>
  void _on(const std::string& term,
           typename std::function<void(const T&)> callback) {
    handlers.resize(std::max(handlers.size(), I + 1));
    handlers[I] = [callback](const void* buf) {
      return callback(*reinterpret_cast<const T*>(buf));
    };
    branches.push_back(term);
    fn_ptrs.push_back(
        const_cast<void*>(reinterpret_cast<const void*>(&handleDispatch<I>)));
  }

  struct Results {
    uintptr_t enc_size;
    uintptr_t dec_buffer_size;
    std::unique_ptr<char[]> dec_buffer;
  };

  template <typename SrcT, typename DstT = SrcT, typename ProtoT = void>
  struct Options {
    using Src = SrcT;
    using Dst = DstT;
    using Proto = ProtoT;

    Src* from;
    Dst* to = nullptr;
    std::string src_path = "";
    std::string tag_path = "";
    bool round_up_size = false;
    bool expect_dec_buffer = false;  // We expect the decode buffer to be used.
  };

  template <typename OptionsT>
  Results transcode(const OptionsT& options) {
    using Src = typename OptionsT::Src;
    using Dst = typename OptionsT::Dst;
    using Proto = typename OptionsT::Proto;

    const PJProtocol* protocol;
    if constexpr (std::is_same_v<Proto, void>) {
      protocol = ctx->plan<Src>(no_tag ? "" : options.tag_path);
    } else {
      protocol = ctx->planProtocol<Proto>();
    }

    ctx->addEncodeFunction<Src>("encode", protocol,
                                no_src_path ? "" : options.src_path);
    ctx->addDecodeFunction<Dst>("decode", protocol, branches);
    ctx->addSizeFunction<Src>("size", protocol,
                              no_src_path ? "" : options.src_path,
                              options.round_up_size);

    const auto portal = ctx->compile();

    const auto size_fn = portal.getSizeFunction<Src>("size");
    const auto encode_fn = portal.getEncodeFunction<Src>("encode");
    const auto decode_fn = portal.getDecodeFunction<Dst>("decode");

    Results results;
    results.enc_size = size_fn(options.from);
    results.dec_buffer_size = 0;
    if (options.to != nullptr) {
      while (true) {
        auto enc_buffer = std::make_unique<char[]>(results.enc_size);
        results.dec_buffer = std::make_unique<char[]>(results.dec_buffer_size);

        encode_fn(options.from, enc_buffer.get());

        auto bbuf = decode_fn(
            enc_buffer.get(), options.to,
            {.ptr = results.dec_buffer.get(), .size = results.dec_buffer_size},
            reinterpret_cast<Handler<Dst>*>(fn_ptrs.data()), &handlers);

        if (bbuf.ptr == nullptr) {
          EXPECT_TRUE(options.expect_dec_buffer);

          // Horribly inefficient, but it ensures that any off-by-one error in
          // the decoding size check will be caught by ASAN.
          results.dec_buffer_size += 1;
          continue;
        }
        if (bbuf.size == results.dec_buffer_size) {
          results.dec_buffer = nullptr;
        }
        break;
      }
    }

    return results;
  }

  std::vector<std::string> branches;
  std::vector<void*> fn_ptrs;
  std::vector<std::function<void(const void*)>> handlers;
  std::set<std::string> waiting_matches;
  bool no_tag;
  bool no_src_path;
  std::unique_ptr<runtime::Context> ctx;
};

using PJVariantTest = PJGenericTest<true>;
using PJTest = PJGenericTest<false>;

}  // namespace pj
