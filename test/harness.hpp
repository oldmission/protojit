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
  void transcode(const X* from, Y* to_msg = nullptr,
                 const std::string& src_path = "",
                 const std::string& tag_path = "") {
    const PJProtocol* protocol = plan<Src>(ctx, no_tag ? "" : tag_path);
    addEncodeFunction<Src>(ctx, "encode", protocol,
                           no_src_path ? "" : src_path);
    addDecodeFunction<Dest>(ctx, "decode", protocol, branches);

    const auto portal = compile(ctx);

    const auto encode =
        portal->ResolveTarget<void (*)(const X*, char*)>("encode");

    const auto decode =
        portal->ResolveTarget<const char* (*)(const char*, char*,
                                              std::pair<char*, uintptr_t>,
                                              const void*)>("decode");

    char buffer[1024];
    encode(from, buffer);
    decode(buffer, reinterpret_cast<char*>(to_msg), std::make_pair(nullptr, 0),
           &handlers);
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
