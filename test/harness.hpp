#pragma once

#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "pj/protojit.hpp"

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

  template <typename Src, typename Dest = Src, typename Proto = void,
            typename X, typename Y = void>
  std::pair<std::unique_ptr<char[]>, uintptr_t> transcode(
      const X* from, Y* to_msg = nullptr, const std::string& src_path = "",
      const std::string& tag_path = "", const CompilationParams& params = {}) {
    const PJProtocol* protocol;
    if constexpr (std::is_same_v<Proto, void>) {
      protocol = plan<Src>(ctx, no_tag ? "" : tag_path);
    } else {
      protocol = planProtocol<Proto>(ctx);
    }

    addEncodeFunction<Src>(ctx, "encode", protocol,
                           no_src_path ? "" : src_path);
    addDecodeFunction<Dest>(ctx, "decode", protocol, branches);
    addSizeFunction<Src>(ctx, "size", protocol, no_src_path ? "" : src_path);

    const auto portal = compile(ctx, params);

    const auto size_fn = portal->ResolveTarget<uintptr_t (*)(const X*)>("size");
    const auto encode_fn =
        portal->ResolveTarget<void (*)(const X*, char*)>("encode");
    const auto decode_fn =
        portal->ResolveTarget<std::pair<const char*, uintptr_t> (*)(
            const char*, char*, std::pair<char*, uintptr_t>, const void*)>(
            "decode");

    const uintptr_t size = size_fn(from);

    std::unique_ptr<char[]> dec_buffer;
    if (to_msg != nullptr) {
      // TODO: retry with larger decode buffer size if needed
      const uintptr_t dec_size = 1024;
      auto enc_buffer = std::make_unique<char[]>(size);
      dec_buffer = std::make_unique<char[]>(dec_size);

      encode_fn(from, enc_buffer.get());

      auto [_, remaining_size] =
          decode_fn(enc_buffer.get(), reinterpret_cast<char*>(to_msg),
                    std::make_pair(dec_buffer.get(), dec_size), &handlers);

      if (remaining_size == dec_size) {
        dec_buffer = nullptr;
      }
    }

    return std::make_pair(std::move(dec_buffer), size);
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
