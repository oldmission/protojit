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
    ctx = getContext();
    if constexpr (VariantTest) {
      std::tie(no_tag, no_src_path) = this->GetParam();
    } else {
      no_tag = false;
      no_src_path = false;
    }
  }

  ~PJGenericTest() {
    EXPECT_TRUE(waiting_matches.empty());
    freeContext(ctx);
  }

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
    branches.push_back(std::make_pair(
        term, reinterpret_cast<const void*>(&handleDispatch<I>)));
  }

  struct Results {
    uintptr_t enc_size;
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
  };

  template <typename OptionsT>
  Results transcode(const OptionsT& options) {
    using Src = typename OptionsT::Src;
    using Dst = typename OptionsT::Dst;
    using Proto = typename OptionsT::Proto;

    const PJProtocol* protocol;
    if constexpr (std::is_same_v<Proto, void>) {
      protocol = plan<Src>(ctx, no_tag ? "" : options.tag_path);
    } else {
      protocol = planProtocol<Proto>(ctx);
    }

    addEncodeFunction<Src>(ctx, "encode", protocol,
                           no_src_path ? "" : options.src_path);
    addDecodeFunction<Dst>(ctx, "decode", protocol, branches);
    addSizeFunction<Src>(ctx, "size", protocol,
                         no_src_path ? "" : options.src_path,
                         options.round_up_size);

    const auto portal = compile(ctx);

    const auto size_fn =
        portal->ResolveTarget<uintptr_t (*)(const Src*)>("size");
    const auto encode_fn =
        portal->ResolveTarget<void (*)(const Src*, char*)>("encode");
    const auto decode_fn =
        portal->ResolveTarget<std::pair<const char*, uintptr_t> (*)(
            const char*, Dst*, std::pair<char*, uintptr_t>, const void*)>(
            "decode");

    const uintptr_t size = size_fn(options.from);

    std::unique_ptr<char[]> dec_buffer;
    if (options.to != nullptr) {
      // TODO: retry with larger decode buffer size if needed
      const uintptr_t dec_size = 1024;
      auto enc_buffer = std::make_unique<char[]>(size);
      dec_buffer = std::make_unique<char[]>(dec_size);

      encode_fn(options.from, enc_buffer.get());

      auto [_, remaining_size] =
          decode_fn(enc_buffer.get(), options.to,
                    std::make_pair(dec_buffer.get(), dec_size), &handlers);

      if (remaining_size == dec_size) {
        dec_buffer = nullptr;
      }
    }

    return Results{.enc_size = size, .dec_buffer = std::move(dec_buffer)};
  }

  std::vector<std::pair<std::string, const void*>> branches;
  std::vector<std::function<void(const void*)>> handlers;
  std::set<std::string> waiting_matches;
  bool no_tag;
  bool no_src_path;
  PJContext* ctx;
};

using PJVariantTest = PJGenericTest<true>;
using PJTest = PJGenericTest<false>;

}  // namespace pj
