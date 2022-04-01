#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "abstract_types.hpp"
#include "arch.hpp"
#include "concrete_types.hpp"
#include "exceptions.hpp"
#include "portal.hpp"
#include "protocol.hpp"
#include "scope.hpp"
#include "tag.hpp"
#include "target.hpp"

namespace pj {

enum class Side {
  kSend,
  kRecv,
};

struct NegotiateOptions {
  bool allow_truncation = true;
};

const Protocol* PlanProtocol(Scope* scope, const ProtoSpec* spec);

const Protocol* Negotiate(  //
    Scope* scope, const ProtoSpec* recv, const ProtoSpec* send,
    NegotiateOptions opts);

void Validate(const Protocol* proto, const ProtoSpec* spec, Side side,
              NegotiateOptions opts);

std::unique_ptr<Portal> Compile(Scope* scope, const PortalSpec* spec);

template <typename T, size_t N, size_t M>
struct ArrayView {
  const uint64_t size;
  const intptr_t offset;

  ArrayView(const T* data, uint64_t size)
      : size(size),
        offset(reinterpret_cast<const char*>(data) -
               reinterpret_cast<const char*>(this)) {}
};

namespace gen {

template <typename T>
struct BuildConcreteType {};

template <typename T, bool is_char>
struct BuildConcreteTypeForIntegral {
  static const CType* Build(Scope* scope) {
    constexpr auto conv =
        is_char ? AIntType::Conversion::kChar
                : (std::is_signed<T>::value ? AIntType::Conversion::kSigned
                                            : AIntType::Conversion::kUnsigned);
    auto* atype = new (scope) AIntType(Bytes(sizeof(T)), conv);
    return new (scope) CIntType(atype, Bytes(sizeof(T)), Bytes(sizeof(T)));
  }
};

#define DEFINE_INT_CTYPE(type, char)                                          \
  template <>                                                                 \
  struct BuildConcreteType<type> : BuildConcreteTypeForIntegral<type, char> { \
  };

DEFINE_INT_CTYPE(uint8_t, false);
DEFINE_INT_CTYPE(uint16_t, false);
DEFINE_INT_CTYPE(uint32_t, false);
DEFINE_INT_CTYPE(uint64_t, false);

DEFINE_INT_CTYPE(int8_t, false);
DEFINE_INT_CTYPE(int16_t, false);
DEFINE_INT_CTYPE(int32_t, false);
DEFINE_INT_CTYPE(int64_t, false);

DEFINE_INT_CTYPE(char, true);
DEFINE_INT_CTYPE(char16_t, true);
DEFINE_INT_CTYPE(char32_t, true);

template <typename E, size_t N>
struct BuildConcreteType<std::array<E, N>> {
  static const CType* Build(Scope* scope) {
    auto* el = BuildConcreteType<E>::Build(scope);
    auto* atype = new (scope) AArrayType(el->abs(), N);
    return new (scope)
        CArrayType(atype, el, el->alignment(), el->total_size() * N);
  }
};

template <typename T>
struct ProtocolHead {};

struct UniqueTermInfo {
  std::string term_name;
  intptr_t term_index;
};

template <typename V, typename T>
struct GetUniqueTermInfo {};

template <typename V>
struct GetTermsForVariant {};

template <typename T, size_t N, size_t M>
struct BuildConcreteType<ArrayView<T, N, M>> {
  static const CType* Build(Scope* scope) {
    auto* el = BuildConcreteType<T>::Build(scope);
    auto* abs = new (scope) AListType(el->abs(), N, M);
    return new (scope) CListType{
        abs,
        /*alignment=*/Bytes(8),
        /*total_size=*/Bytes(16),
        el,
        /*ref_offset=*/Bytes(8),
        /*ref_size=*/Bytes(8),
        /*partial_payload_offset=*/Width::None(),
        /*partial_payload_count=*/0,
        /*full_payload_offset=*/Width::None(),
        /*full_payload_count=*/0,
        /*len_offset=*/Bytes(0),
        /*len_size=*/Bytes(8),
    };
  }
};

}  // namespace gen

}  // namespace pj
