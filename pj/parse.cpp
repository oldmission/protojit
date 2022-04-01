#include <charconv>
#include <pegtl.hpp>
#include <pegtl/contrib/analyze.hpp>

#include "protogen.hpp"
#include "protojit.hpp"

using namespace tao::pegtl;

namespace pj {

struct ParseState {
  ParsingScope& parse_scope;
  Scope& scope;

  std::vector<std::filesystem::path>& imports;
  std::vector<ParsedProtoFile::Decl>& decls;

  // Updated after parsing a 'FieldDecl' rule.
  // Cleared when done with a struct/variant decl.
  std::map<std::string, const AType*> fields;

  // Populated after parsing a `VariantFieldDecl` or `EnumFieldDecl` rule.
  // Cleared after parsing a `VariantDecl` or `EnumDecl` rule.
  std::map<std::string, uint8_t> explicit_tags;

  // Used when parsing int types and modifiers.
  // Populated after parsing a 'Type' rule.
  const AType* type = nullptr;

  // Populated after parsing Len, MinLen, and MaxLen rules.
  intptr_t array_len = kNone;
  intptr_t array_min_len = kNone;
  intptr_t array_max_len = kNone;

  // Set by ExternalDecl, cleared by StructDecl.
  bool is_external = false;

  // Populated by FieldDecl, cleared by StructDecl.
  std::vector<std::string> field_order;

  // Populated after parsing ExplicitTagDecl.
  // Cleared after parsing VariantFieldDecl or EnumFieldDecl.
  //
  // An explicit tag cannot be 0, since that is reserved for
  // the undefined case. Thus, 0 means no explicit tag was
  // specified.
  uint8_t explicit_tag = 0;

  template <typename T, typename... Args>
  T* New(Args&&... args) {
    return new (scope) T(std::forward<Args>(args)...);
  }

  SourceId space;

  // Populated after parsing Id and ScopedId.
  // Cleared by PopId() and PopScopedId().
  std::vector<std::vector<std::string>> ids = {{}};

  std::string PopId() {
    assert(ids.size() > 1);
    assert(ids[ids.size() - 2].size() == 1);
    auto s = ids[ids.size() - 2][0];

    std::swap(ids[ids.size() - 2], ids.back());
    ids.pop_back();

    return s;
  }

  SourceId PopScopedId(bool include_space = true) {
    SourceId result = ids[ids.size() - 2];

    std::swap(ids[ids.size() - 2], ids.back());
    ids.pop_back();

    if (!include_space) {
      return result;
    }

    auto s = space;
    for (auto& p : result) {
      s.push_back(p);
    }
    return s;
  }

  template <typename I>
  const AType* ResolveType(const I& in, const SourceId& id) {
    // TODO: this is super inefficient!
    // Use twine or iterators instead.
    for (intptr_t i = space.size(); i >= 0; --i) {
      SourceId id_;
      for (intptr_t j = 0; j < i; ++j) {
        id_.push_back(space[j]);
      }
      for (auto& p : id) {
        id_.emplace_back(p);
      }

      if (auto it = parse_scope.type_defs.find(id_);
          it != parse_scope.type_defs.end()) {
        return it->second;
      }
    }
    // for (auto& [k, v] : parse_scope.type_defs) {
    //   std::cerr << "! ";
    //   for (auto& p : k) {
    //     std::cerr << p << " ";
    //   }
    //   std::cerr << " -> ";
    //   for (auto& p : v->name) {
    //     std::cerr << p << " ";
    //   }
    //   std::cerr << "\n";
    // }
    throw parse_error("Cannot resolve ID.", in.position());
  }

  template <typename I>
  void DefineType(const I& in, const SourceId& id, const AType* type) {
    if (parse_scope.type_defs.count(id)) {
      throw parse_error("Cannot re-define type", in.position());
    }
    parse_scope.type_defs.emplace(std::move(id), type->AsNamed());
  }
};

#define __ state->

template <typename Rule>
struct ParseAction : nothing<Rule> {};

#define BEGIN_ACTION(R)             \
  template <>                       \
  struct ParseAction<R> {           \
    template <typename ActionInput> \
    static void apply(const ActionInput& in, ParseState* state) {
#define END_ACTION() \
  }                  \
  }                  \
  ;

struct num : plus<digit> {};

struct space_or_comment
    : sor<space, seq<string<'/', '/'>, star<seq<not_at<eol>, any>>, eol>> {};

template <typename T>
struct spaced : pad<T, space_or_comment> {};

template <char C>
struct tok : pad<string<C>, space_or_comment> {};

// Naming these up here since this syntax often screws up the
// editor's highlighting.
using LB = tok<'{'>;
using RB = tok<'}'>;

#define KEYWORD(X) spaced<TAO_PEGTL_KEYWORD(X)>

struct struct_key : pad<TAO_PEGTL_KEYWORD("struct"), space> {};
struct variant_key : pad<TAO_PEGTL_KEYWORD("variant"), space> {};

template <intptr_t kPrefix, AIntType::Conversion conv, typename ActionInput>
static void parse_int(const ActionInput& in, ParseState* state) {
  assert(in.size() > kPrefix);
  const char* num_start = in.begin() + kPrefix;
  intptr_t bits;
  DEBUG_ONLY(auto result =) std::from_chars(num_start, in.end(), bits);
  assert(result.ptr == in.end());
  // SAMIR_TODO: validate bits
  __ type = __ New<AIntType>(Bits(bits), conv);
}

struct UIntType : seq<string<'u', 'i', 'n', 't'>, num> {};

BEGIN_ACTION(UIntType) {
  parse_int<4, AIntType::Conversion::kUnsigned>(in, state);
}
END_ACTION()

struct IntType : seq<string<'i', 'n', 't'>, num> {};

BEGIN_ACTION(IntType) {
  parse_int<3, AIntType::Conversion::kSigned>(in, state);
}
END_ACTION()

struct CharType : seq<string<'c', 'h', 'a', 'r'>, num> {};

BEGIN_ACTION(CharType) { parse_int<4, AIntType::Conversion::kChar>(in, state); }
END_ACTION()

struct Identifier : identifier {};

BEGIN_ACTION(Identifier) { __ ids.back().emplace_back(in.string_view()); }
END_ACTION()

struct EndId : seq<> {};

BEGIN_ACTION(EndId) { __ ids.emplace_back(); }
END_ACTION()

struct Id : seq<Identifier, EndId> {};

struct IdSuffix : if_must<tok<'.'>, Identifier> {};

struct ScopedId : must<Identifier, star<IdSuffix>, EndId> {};

struct TypeRef : ScopedId {};

BEGIN_ACTION(TypeRef) {
  assert(__ type == nullptr);
  auto id = __ PopScopedId(false);
  __ type = __ ResolveType(in, id);
}
END_ACTION()

struct NonArrayType : sor<UIntType, IntType, CharType, TypeRef> {};

struct Len : num {};
struct FixedArrayModifier : seq<tok<'['>, Len, tok<']'>> {};

BEGIN_ACTION(Len) {
  intptr_t length;
  DEBUG_ONLY(auto result =) std::from_chars(in.begin(), in.end(), length);
  assert(result.ptr == in.end());
  __ array_len = length;
}
END_ACTION()

BEGIN_ACTION(FixedArrayModifier) {
  assert(__ array_len != kNone);
  assert(__ type != nullptr);
  __ type = __ New<AArrayType>(__ type, __ array_len);
  __ array_len = kNone;
}
END_ACTION()

struct MinLen : num {};

BEGIN_ACTION(MinLen) {
  intptr_t length;
  DEBUG_ONLY(auto result =) std::from_chars(in.begin(), in.end(), length);
  assert(result.ptr == in.end());
  __ array_min_len = length;
}
END_ACTION()

struct MaxLen : num {};

BEGIN_ACTION(MaxLen) {
  intptr_t length;
  DEBUG_ONLY(auto result =) std::from_chars(in.begin(), in.end(), length);
  assert(result.ptr == in.end());
  __ array_max_len = length;
}
END_ACTION()

struct VarArrayModifier
    : seq<tok<'['>, opt<MinLen>, tok<':'>, opt<MaxLen>, tok<']'>> {};

BEGIN_ACTION(VarArrayModifier) {
  assert(__ type != nullptr);
  __ type = __ New<AListType>(__ type, __ array_min_len, __ array_max_len);
  __ array_max_len = kNone;
  __ array_min_len = kNone;
}
END_ACTION()

struct ArrayModifier : sor<FixedArrayModifier, VarArrayModifier> {};

struct Type : seq<NonArrayType, star<ArrayModifier>> {};

struct FieldDecl : if_must<Id, tok<':'>, Type, tok<';'>> {};

BEGIN_ACTION(FieldDecl) {
  auto field_name = __ PopId();
  assert(__ type != nullptr);
  __ fields[field_name] = __ type;
  __ type = nullptr;
  __ field_order.push_back(field_name);
}
END_ACTION()

struct TopDecl;

struct TypeDecl : if_must<KEYWORD("type"), Id, tok<'='>, Type, tok<';'>> {};

BEGIN_ACTION(TypeDecl) {
  assert(__ type != nullptr);
  auto name = __ PopScopedId();
  auto type = __ New<ANamedType>(SourceId(name), __ type);
  __ DefineType(in, name, type);

  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kType,
      .name = std::move(name),
      .atype = type,
      .ctype = nullptr,
  });
  __ type = nullptr;
}
END_ACTION();

struct ExternalDecl : KEYWORD("external") {};

BEGIN_ACTION(ExternalDecl) { __ is_external = true; }
END_ACTION()

struct StructDecl : if_must<KEYWORD("struct"), Id, opt<ExternalDecl>, LB,
                            star<FieldDecl>, RB> {};

BEGIN_ACTION(StructDecl) {
  decltype(ParseState::fields) fields;
  std::swap(fields, __ fields);
  auto name = __ PopScopedId();
  auto type = __ New<ANamedType>(SourceId(name),
                                 __ New<AStructType>(std::move(fields)));
  __ DefineType(in, name, type);
  decltype(__ field_order) field_order;
  std::swap(field_order, __ field_order);

  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kComposite,
      .name = std::move(name),
      .atype = type,
      .ctype = nullptr,
      .field_order = std::move(field_order),
      .is_external = __ is_external,
  });

  __ is_external = false;
}
END_ACTION()

struct ExplicitTag : sor<num, seq<string<'\''>, any, string<'\''>>> {};

BEGIN_ACTION(ExplicitTag) {
  if (*in.begin() == '\'') {
    __ explicit_tag = in.begin()[1];
  } else {
    DEBUG_ONLY(auto result =)
    std::from_chars(in.begin(), in.end(), __ explicit_tag);
    assert(result.ptr == in.end());
  }
  if (__ explicit_tag == 0) {
    throw parse_error("Cannot use 0 as a tag value (reserved for undefined).\n",
                      in.position());
  }
}
END_ACTION()

struct ExplicitTagDecl : opt<if_must<tok<'='>, ExplicitTag>> {};

struct VariantFieldDecl
    : if_must<Id, opt<tok<':'>, Type>, ExplicitTagDecl, tok<';'>> {};

BEGIN_ACTION(VariantFieldDecl) {
  auto field_name = __ PopId();
  // The type may be null, indicating no payload is attached.
  __ fields[field_name] = __ type;
  __ type = nullptr;
  if (__ explicit_tag) {
    __ explicit_tags[field_name] = __ explicit_tag;
    __ explicit_tag = 0;
  }
}
END_ACTION()

template <typename ActionInput>
static void HandleVariant(const ActionInput& in, ParseState* state,
                          bool is_enum) {
  decltype(ParseState::fields) terms;
  std::swap(terms, __ fields);

  decltype(ParseState::explicit_tags) tags;
  std::swap(tags, __ explicit_tags);

  auto name = __ PopScopedId();
  auto type = __ New<ANamedType>(std::move(name),
                                 __ New<AVariantType>(std::move(terms)));
  __ DefineType(in, name, type);
  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kComposite,
      .name = std::move(name),
      .atype = type,
      .ctype = nullptr,
      .is_enum = is_enum,
      .explicit_tags = std::move(tags),
  });
}

struct VariantDecl
    : if_must<KEYWORD("variant"), Id, LB, star<VariantFieldDecl>, RB> {};

BEGIN_ACTION(VariantDecl) { HandleVariant(in, state, /*is_enum=*/false); }
END_ACTION()

struct EnumFieldDecl : if_must<Id, ExplicitTagDecl, tok<';'>> {};

BEGIN_ACTION(EnumFieldDecl) {
  auto field_name = __ PopId();
  __ fields[field_name] = nullptr;
  if (__ explicit_tag) {
    __ explicit_tags[field_name] = __ explicit_tag;
    __ explicit_tag = 0;
  }
}
END_ACTION()

struct EnumDecl : if_must<KEYWORD("enum"), Id, LB, star<EnumFieldDecl>, RB> {};

BEGIN_ACTION(EnumDecl) { HandleVariant(in, state, /*is_enum=*/true); }
END_ACTION()

struct ScopeBegin : LB {};
BEGIN_ACTION(ScopeBegin) {
  auto name = __ PopId();
  __ space.push_back(name);
}
END_ACTION()

struct ScopeEnd : RB {};
BEGIN_ACTION(ScopeEnd) { __ space.pop_back(); }
END_ACTION()

struct SpaceDecl
    : if_must<KEYWORD("space"), Id, ScopeBegin, star<TopDecl>, ScopeEnd> {};

struct ImportDecl : if_must<KEYWORD("import"), ScopedId, tok<';'>> {};

BEGIN_ACTION(ImportDecl) {
  auto id = __ PopScopedId();

  std::filesystem::path found_path;
  bool found = false;

  for (auto path : __ parse_scope.import_dirs) {
    for (auto& p : id) {
      path /= p;
    }
    path += ".pj";
    if (std::filesystem::exists(path)) {
      found_path = path;
      found = true;
    }
  }

  if (found) {
    ParseProtoFile(__ parse_scope, found_path);
    __ imports.emplace_back(found_path);
  } else {
    throw parse_error("Cannot find import", in.position());
  }
}
END_ACTION()

struct TopDecl
    : sor<ImportDecl, StructDecl, VariantDecl, EnumDecl, /*ProtoDecl,*/
          TypeDecl, SpaceDecl> {};

struct ParseFile : must<star<TopDecl>, eof> {};

void ParseProtoFile(ParsingScope& scope, const std::filesystem::path& path) {
  if (scope.pending_files.count(path)) {
    std::cerr << "Cycle in imports:\n";
    for (auto& p : scope.stack) {
      std::cerr << "  " << p << "\n";
      if (p == path) break;
    }
    internal::iterator iterator(nullptr, 0, 0, 0);
    throw parse_error("Cycle in imports!", position(iterator, path));
  }

  scope.stack.push_back(path);
  scope.pending_files.emplace(path);

  auto& parsed = scope.parsed_files[path];
  ParseState state{
      .parse_scope = scope,
      .scope = scope.scope,
      .imports = parsed.imports,
      .decls = parsed.decls,
  };
  file_input in(path);
  parse<ParseFile, ParseAction>(in, &state);

  scope.stack.pop_back();
  scope.pending_files.erase(path);
}

}  // namespace pj
