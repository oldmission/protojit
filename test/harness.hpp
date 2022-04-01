#include <gtest/gtest.h>
#include <llvm/Support/Debug.h>

#include <functional>

#include "pj/protojit.hpp"

namespace pj {

template <size_t I>
static void HandleDispatch(
    const void* received,
    std::vector<std::function<void(const void*)>>* state) {
  return (*state)[I](received);
}

class PJTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  ~PJTest() { EXPECT_TRUE(waiting_matches.empty()); }

  template <size_t I, typename T>
  void OnMatch(const std::string& term,
               typename std::function<void(const T&)> callback) {
    waiting_matches.insert(term);
    _On<I, T>(term, [this, term, callback = std::move(callback)](const T& t) {
      waiting_matches.erase(term);
      callback(t);
    });
  }

  template <size_t I, typename T>
  void OnNoMatch(const std::string& term,
                 typename std::function<void(const T&)> callback) {
    _On<I, T>(term, [this, term, callback = std::move(callback)](const T& t) {
      waiting_matches.insert(term);
      callback(t);
    });
  }

  template <size_t I, typename T>
  void _On(const std::string& term,
           typename std::function<void(const T&)> callback) {
    handlers.resize(std::max(handlers.size(), I + 1));
    handlers[I] = [callback](const void* value) {
      return callback(*reinterpret_cast<const T*>(value));
    };

    branches[term] = reinterpret_cast<const void*>(&HandleDispatch<I>);
  }

  template <typename F, typename T = F, typename M = F, typename X,
            typename Y = void>
  void Transcode(const X* from, Y* to_msg = nullptr, Path&& encode_path = {},
                 Path&& tag_path = {}, Path&& decode_path = {},
                 ProtoParams params = {}) {
    const auto* from_type = gen::BuildConcreteType<F>::Build(&S);
    const auto* through_type = gen::BuildConcreteType<M>::Build(&S);
    const auto* to_type = gen::BuildConcreteType<T>::Build(&S);

    const auto* spec =
        new (S) ProtoSpec(std::move(tag_path), through_type, params);
    const auto* protocol = PlanProtocol(&S, spec);

    std::vector<const pj::Target*> targets;
    targets.emplace_back(
        new (S) pj::EncodeTarget("encode", encode_path, from_type, protocol));

    const Width total_size = protocol->SizeOf(from_type, encode_path);
    ASSERT_TRUE(total_size.IsNotNone());

    std::unique_ptr<char[]> to_buf;
    char* to;
    if (to_msg != nullptr) {
      to = reinterpret_cast<char*>(to_msg);
    } else {
      to_buf.reset(new char[sizeof(T)]);
      to = to_buf.get();
    }

    targets.emplace_back(new (S) pj::DecodeTarget("decode", protocol, to_type,
                                                  std::move(decode_path),
                                                  std::move(branches)));

    const auto* portal_spec = new (S) pj::PortalSpec(std::move(targets));
    const auto portal = pj::Compile(&S, portal_spec);

    const auto buf = std::make_unique<char[]>(total_size.bytes());
    memset(buf.get(), 0xcc, total_size.bytes());

    const auto encode =
        portal->ResolveTarget<void (*)(const X*, char*)>("encode");
    const auto decode = portal->ResolveTarget<void (*)(
        const void*, const char*, const char*, size_t)>("decode");

    encode(from, buf.get());
    decode(&handlers, buf.get(), to, total_size.bytes());
  }

  template <typename M, typename T = M>
  auto GenSize(intptr_t max_proto_size = kNone, Path path = {},
               Path tag_path = {}) -> std::function<intptr_t(const void*)> {
    const auto* spec =
        new (S) ProtoSpec(tag_path, gen::BuildConcreteType<T>::Build(&S),
                          {.max_size = Bytes(max_proto_size)});
    const auto* protocol = PlanProtocol(&S, spec);

    auto static_size = protocol->SizeOf(nullptr, path);
    if (static_size.IsNotNone()) {
      return [=](const void*) { return static_size.bytes(); };
    }

    std::vector<const pj::Target*> targets;
    targets.emplace_back(new (S) pj::SizeTarget(
        "sizeof", {}, gen::BuildConcreteType<M>::Build(&S), protocol));

    const auto* portal_spec = new (S) pj::PortalSpec(std::move(targets));
    auto portal = pj::Compile(&S, portal_spec);
    auto result = portal->ResolveTarget<intptr_t (*)(const void*)>("sizeof");
    portals.push_back(std::move(portal));
    return result;
  }

  std::set<std::string> waiting_matches;
  std::map<std::string, const void*> branches;
  std::vector<std::function<void(const void*)>> handlers;
  std::vector<std::unique_ptr<Portal>> portals;
  pj::Scope S;
};

}  // namespace pj
