#pragma once

#include <cassert>
#include <map>
#include <set>
#include <unordered_map>

#include "arch.hpp"
#include "scope.hpp"

namespace pj {

class CType;
struct ParsedProtoFile;

using Path = std::vector<std::string>;

#define FOR_EACH_TYPE(V) \
  V(Any)                 \
  V(Array)               \
  V(Int)                 \
  V(List)                \
  V(Named)               \
  V(Outlined)            \
  V(Struct)              \
  V(Variant)

#define DECLARE(name) class A##name##Type;
FOR_EACH_TYPE(DECLARE)
#undef DECLARE

class AType : public Scoped {
 public:
  AType() {}
  virtual ~AType();

  virtual void Validate() const = 0;

  const CType* Plan(
      Scope* S, std::unordered_map<const AType*, const CType*>& memo) const {
    if (auto it = memo.find(this); it != memo.end()) {
      return it->second;
    }
    auto* ctype = PlanMemo(S, memo);
    memo[this] = ctype;
    return ctype;
  }

  virtual const CType* PlanMemo(
      Scope*, std::unordered_map<const AType*, const CType*>& memo) const = 0;

#define HANDLE_TYPE(name)                         \
  virtual bool Is##name() const { return false; } \
  virtual const A##name##Type* As##name() const { \
    throw std::logic_error("AType cast");         \
  }

  FOR_EACH_TYPE(HANDLE_TYPE)
#undef HANDLE_TYPE
};

#define MAKE_ABS(name)                                                        \
 public:                                                                      \
  ~A##name##Type() override;                                                  \
  bool Is##name() const override { return true; }                             \
  const A##name##Type* As##name() const override { return this; }             \
  const CType* PlanMemo(Scope*,                                               \
                        std::unordered_map<const AType*, const CType*>& memo) \
      const override;                                                         \
  void Validate() const override;

class AIntType : public AType {
 public:
  enum class Conversion {
    kSigned,
    kUnsigned,
    kChar,
  };

  const Width len;
  const Conversion conv;

  MAKE_ABS(Int);

  AIntType(Width len, Conversion conv) : len(len), conv(conv) {}
};

class AAnyType : public AType {
  MAKE_ABS(Any);
};

class AVariantType : public AType {
 public:
  const std::map<std::string, const AType*> terms;
  const std::map<std::string, std::set<std::string>> aliases;

  AVariantType(decltype(terms)&& terms) : terms(terms) {
    assert(!terms.count("undef"));
  }

  AVariantType(decltype(terms)&& terms, decltype(aliases)&& aliases)
      : terms(terms), aliases(aliases) {
    assert(!terms.count("undef"));
  }

  MAKE_ABS(Variant);

  const CType* PlanWithTags(
      Scope*, std::unordered_map<const AType*, const CType*>& memo,
      std::map<std::string, uint8_t> explicit_tags) const;
};

class AStructType : public AType {
 public:
  const std::map<std::string, const AType*> fields;
  const std::map<std::string, std::set<std::string>> aliases;

  AStructType(std::map<std::string, const AType*>&& fields) : fields(fields) {}

  AStructType(std::map<std::string, const AType*>&& fields,
              std::map<std::string, std::set<std::string>>&& aliases)
      : fields(fields), aliases(aliases) {}

  const CType* PlanWithFieldOrder(
      Scope*, std::unordered_map<const AType*, const CType*>& memo,
      const std::vector<std::string>& explicit_tags) const;

  MAKE_ABS(Struct);
};

class AArrayType : public AType {
 public:
  const AType* const el;
  const intptr_t length;

  AArrayType(const AType* el, intptr_t length) : el(el), length(length) {}

  MAKE_ABS(Array);
};

class AListType : public AType {
 public:
  const AType* const el;

  // Either can be kNone.
  const intptr_t min_len = kNone;
  const intptr_t max_len = kNone;

  AListType(const AType* el, intptr_t min_len, intptr_t max_len)
      : el(el), min_len(min_len), max_len(max_len) {}

  MAKE_ABS(List);
};

class AOutlinedType : public AType {
 public:
  AType* const el;
  const bool optional;

  AOutlinedType(AType* el, bool optional) : el(el), optional(optional) {}

  MAKE_ABS(Outlined)
};

class ANamedType : public AType {
 public:
  // const SourceId name;
  const AType* const named;

  ANamedType(/*SourceId&& name, */const AType* named) : named(named) {
    assert(named);
  }

  MAKE_ABS(Named)
};

#undef MAKE_ABS

}  // namespace pj
