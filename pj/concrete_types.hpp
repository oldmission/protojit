#pragma once

#include "abstract_types.hpp"

namespace mlir {
class Type;
class MLIRContext;
class OpBuilder;
class Location;
class Operation;
class Value;
};  // namespace mlir

namespace llvm {
template <typename T>
class ArrayRef;
class StringRef;
};  // namespace llvm

namespace pj {

class DecodeTarget;
struct ProtoParams;

struct PathPiece {
  Path::const_iterator begin;
  Path::const_iterator end;

  PathPiece() {}
  PathPiece(Path path) : begin(path.begin()), end(path.end()) {}
  PathPiece(decltype(begin) begin, decltype(end) end)
      : begin(begin), end(end) {}
};

#define DECLARE(name) class C##name##Type;
FOR_EACH_TYPE(DECLARE)
#undef DECLARE

class CType : public Scoped {
 public:
  CType(const AType* abs, Width alignment, Width total_size)
      : abs_(abs), alignment_(alignment), total_size_(total_size) {
    assert(alignment_.IsBytes() && total_size_.IsBytes());
  }
  virtual ~CType();

#define HANDLE_TYPE(name)                                                     \
  virtual bool Is##name() const { return false; }                             \
  virtual C##name##Type* As##name() { throw std::logic_error("CType cast"); } \
  virtual const C##name##Type* As##name() const {                             \
    throw std::logic_error("CType cast");                                     \
  }
  FOR_EACH_TYPE(HANDLE_TYPE)
#undef HANDLE_TYPE

  const AType* abs() const { return abs_; }

  void ValidateHead() const { Validate(true); }
  virtual void Validate(bool has_tag) const = 0;
  virtual const CType* Plan(Scope&, const ProtoParams& params,
                            PathPiece tag) const = 0;

  virtual mlir::Value GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,
                              const CType* cto, mlir::OpBuilder& builder,
                              const mlir::Value& source) const = 0;

  virtual mlir::Value GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,
                                PathPiece path, const CType* cto,
                                mlir::OpBuilder& builder,
                                const mlir::Value& from,
                                const mlir::Value& to) const = 0;

  virtual void GenDecode(Scope* S, mlir::MLIRContext* C,
                         const DecodeTarget& target, PathPiece tag,
                         PathPiece dispatch, const CType* cto,
                         mlir::OpBuilder& builder, const mlir::Value& base,
                         const mlir::Value& tagv, const mlir::Value& from,
                         const mlir::Value& to,
                         const mlir::Value& state) const = 0;

  virtual mlir::Value GenDefault(mlir::MLIRContext* C,
                                 const mlir::Location& loc,
                                 mlir::OpBuilder& builder,
                                 const mlir::Value& to) const = 0;

  Width alignment() const { return alignment_; }
  Width total_size() const { return total_size_; }

  Width aligned_size() const { return RoundUp(total_size_, alignment_); }

  mlir::Type toIR(mlir::MLIRContext* C) const;
  virtual Width ImpliedSize(const CType* from, PathPiece path,
                            PathPiece tag) const {
    return total_size();
  }

  virtual const CType* Resolve(PathPiece tag) const { return nullptr; }

 private:
  const AType* const abs_;
  const Width alignment_;
  const Width total_size_;
};

#define MAKE_CON(name)                                                        \
 public:                                                                      \
  bool Is##name() const override { return true; }                             \
  C##name##Type* As##name() override { return this; }                         \
  const C##name##Type* As##name() const override { return this; }             \
  const A##name##Type* abs() const { return this->CType::abs()->As##name(); } \
  virtual const CType* Plan(Scope&, const ProtoParams& params, PathPiece tag) \
      const override;                                                         \
  void Validate(bool has_tag) const override;                                 \
                                                                              \
  mlir::Value GenSize(Scope* S, mlir::MLIRContext* C, PathPiece path,         \
                      const CType* cto, mlir::OpBuilder& builder,             \
                      const mlir::Value& source) const override;              \
                                                                              \
  mlir::Value GenEncode(Scope* S, mlir::MLIRContext* C, PathPiece tag,        \
                        PathPiece path, const CType* cto,                     \
                        mlir::OpBuilder& builder, const mlir::Value& from,    \
                        const mlir::Value& to) const override;                \
                                                                              \
  void GenDecode(Scope* S, mlir::MLIRContext* C, const DecodeTarget& target,  \
                 PathPiece tag, PathPiece dispatch, const CType* cto,         \
                 mlir::OpBuilder& builder, const mlir::Value& base,           \
                 const mlir::Value& tagv, const mlir::Value& from,            \
                 const mlir::Value& to, const mlir::Value& state)             \
      const override {                                                        \
    return GenDecode(S, C, target, tag, dispatch, cto->As##name(), builder,   \
                     base, tagv, from, to, state);                            \
  }                                                                           \
                                                                              \
  void GenDecode(Scope* S, mlir::MLIRContext* C, const DecodeTarget& target,  \
                 PathPiece tag, PathPiece dispatch, const C##name##Type* cto, \
                 mlir::OpBuilder& builder, const mlir::Value& base,           \
                 const mlir::Value& tagv, const mlir::Value& from,            \
                 const mlir::Value& to, const mlir::Value& state) const;      \
                                                                              \
  mlir::Value GenDefault(mlir::MLIRContext* C, const mlir::Location& loc,     \
                         mlir::OpBuilder& builder, const mlir::Value& to)     \
      const override;

class CStructType : public CType {
 public:
  struct CStructField {
    Width offset;
    const CType* type;
  };

  const std::map<std::string, CStructField> fields;

  CStructType(const AType* abs, Width alignment, Width total_size,
              std::map<std::string, CStructField>&& fields)
      : CType(abs, alignment, total_size), fields(fields) {}

  const CType* Resolve(PathPiece) const override;
  Width ImpliedSize(const CType* from, PathPiece path,
                    PathPiece tag) const override;

  MAKE_CON(Struct);

  friend class AStructType;
};

class CVariantType : public CType {
 public:
  static constexpr intptr_t kUndefTag = 0;

  struct CTerm {
    const intptr_t tag;
    const CType* const type;
  };

  const std::map<std::string, CTerm> terms;

  const Width term_offset;
  const Width term_size;

  // Can be -1 if the tag is only available through the
  // joint tag. This requires that the variant's tag is
  // available in the joint tag.
  // TODO(8): validate that the tag comes before the term
  // when the total size is unknown.
  const Width tag_offset;
  const Width tag_size;

  Width ImpliedSize(const CType* from, PathPiece path,
                    PathPiece tag) const override;

  void GenEncodeForTerm(pj::Scope* S, mlir::MLIRContext* C, PathPiece tag,
                        const std::string& term, PathPiece path,
                        const CVariantType* cto, mlir::OpBuilder& _,
                        const mlir::Value& from, const mlir::Value& to) const;

  void GenDecodeForTerm(pj::Scope* S, mlir::MLIRContext* C,
                        const DecodeTarget& target, PathPiece tag,
                        PathPiece dispatch, const std::string& head,
                        const CVariantType* cto, mlir::OpBuilder& _,
                        const mlir::Value& base, const mlir::Value& tagv,
                        const mlir::Value& from, const mlir::Value& to,
                        const mlir::Value& state) const;

  // Total_size (and term_size) can be kNone for CVariantType, provided
  // tag_offset is also kNone.
  CVariantType(const AVariantType* abs, Width alignment, Width total_size,
               decltype(terms)&& terms, Width term_offset, Width term_size,
               Width tag_offset, Width tag_size);

  const CVariantType* RemoveTag(Scope* scope) const;

  bool MatchesExactlyAsEnum(const CVariantType* other) const;

  mlir::Type GetTagAsIntegerType(Scope* scope, mlir::MLIRContext* C) const;

  const CType* Resolve(PathPiece) const override;

  MAKE_CON(Variant);
};

class CArrayType : public CType {
  const CType* const el_;

 public:
  CArrayType(const AArrayType* abs, const CType* el, Width alignment,
             Width total_size)
      : CType(abs, alignment, total_size), el_(el) {}

  const CType* el() const { return el_; }

  MAKE_CON(Array);
};

class CIntType : public CType {
 public:
  CIntType(const AIntType* abs, Width alignment, Width total_size)
      : CType(abs, alignment, total_size) {}

  MAKE_CON(Int);
};

class CListType : public CType {
 public:
  const CType* const el;

  const Width len_offset;
  const Width len_size;

  // Can be -1 if overflow is not possible.
  // Can overlap with payload.
  const Width ref_offset;
  const Width ref_size;

  const intptr_t partial_payload_count;

  // Can be kNone if it will never overflow,
  // or there is no partial payload on overflow
  // (partial_payload_count = 0).
  const Width partial_payload_offset;

  // Can be kNone if it will always overflow.
  const Width full_payload_offset;
  const intptr_t full_payload_count;

  CListType(const AListType* abs, Width alignment, Width total_size,
            const CType* el, Width ref_offset, Width ref_size,
            Width partial_payload_offset, intptr_t partial_payload_count,
            Width full_payload_offset, intptr_t full_payload_count,
            Width len_offset, Width len_size);

  MAKE_CON(List);

  mlir::Value LoadLength(mlir::Location loc, mlir::Value value,
                         mlir::OpBuilder& _) const;
  mlir::Value LoadOutlinedArray(mlir::Location loc, mlir::Value value,
                                mlir::Type inner, mlir::OpBuilder& _) const;

  Width ImpliedSize(const CType* from, PathPiece path,
                    PathPiece tag) const override {
    return ref_offset.IsNone() ? total_size() : Width::None();
  }
};

class CAnyType : public CType {
 public:
  // Data is not guaranteed to have any specific alignment.
  const intptr_t data_offset;
  const intptr_t data_size;

  const intptr_t type_offset;
  const intptr_t type_size;

  // These can both be 0 if no type id is used.
  const intptr_t type_id_offset;
  const intptr_t type_id_size;

  CAnyType(AAnyType* abs, Width alignment, Width total_size,
           intptr_t data_offset, intptr_t data_size, intptr_t type_offset,
           intptr_t type_size, intptr_t type_id_offset, intptr_t type_id_size);

  MAKE_CON(Any);
};

class COutlinedType : public CType {
 public:
  const intptr_t ref_size;

  COutlinedType(AOutlinedType* abs, Width alignment, Width total_size,
                intptr_t ref_size)
      : CType(abs, alignment, total_size), ref_size(ref_size) {}

  MAKE_CON(Outlined);
};

class CNamedType : public CType {
 public:
  virtual ~CNamedType();

  const CType* named;

  CNamedType(const ANamedType* abs, const CType* named)
      : CType(abs, named->alignment(), named->total_size()), named(named) {}

  MAKE_CON(Named);
};

#undef MAKE_CON

}  // namespace pj
