#include <mlir/IR/MLIRContext.h>
#include <charconv>
#include <pegtl.hpp>
#include <pegtl/contrib/analyze.hpp>

#include "arch.hpp"
#include "protogen.hpp"
#include "span.hpp"
#include "types.hpp"
#include "validate.hpp"

#include <unordered_set>

using namespace tao::pegtl;

namespace pj {

using SourceId = std::vector<std::string>;

struct ParseState {
  ParsingScope& parse_scope;
  ProtoJitContext& ctx;

  std::vector<std::filesystem::path>& imports;
  std::vector<ParsedProtoFile::Decl>& decls;

  // Updated after parsing a 'FieldDecl' rule.
  // Cleared when done with a struct/variant decl.
  std::map<std::string, pj::types::ValueType> fields;

  // Populated after parsing a `VariantFieldDecl` or `EnumFieldDecl` rule.
  // Cleared after parsing a `VariantDecl` or `EnumDecl` rule.
  std::map<std::string, uint64_t> explicit_tags;

  // Used when parsing int types and modifiers.
  // Populated after parsing a 'Type' rule.
  types::ValueType type;

  // Populated after parsing Len, MinLen, and MaxLen rules.
  uint64_t array_len = kNone;
  int64_t array_min_len = kNone;
  int64_t array_max_len = kNone;

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
  uint64_t explicit_tag = 0;

  // Populated after parsing TagPathDecl.
  // Cleared after parsing ProtoDecl.
  types::PathAttr tag_path;

  SourceId space;

  // Populated after parsing Id and ScopedId.
  // Cleared by popId() and popScopedId().
  std::vector<std::vector<std::string>> ids = {{}};

  std::string popId() {
    assert(ids.size() > 1);
    assert(ids[ids.size() - 2].size() == 1);
    auto s = ids[ids.size() - 2][0];

    std::swap(ids[ids.size() - 2], ids.back());
    ids.pop_back();

    return s;
  }

  SourceId popScopedId(bool include_space = true) {
    SourceId result = std::move(ids[ids.size() - 2]);

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
  types::ValueType resolveType(const I& in, const SourceId& id,
                               bool error_on_failure = true) {
    std::vector<llvm::StringRef> id_vec;
    id_vec.reserve(space.size() + id_vec.size());
    for (intptr_t i = space.size(); i >= 0; --i) {
      for (intptr_t j = 0; j < i; ++j) {
        id_vec.push_back(space[j]);
      }
      for (auto& p : id) {
        id_vec.push_back(p);
      }

      if (auto it = parse_scope.type_defs.find(id_vec);
          it != parse_scope.type_defs.end()) {
        return it->second;
      }
      id_vec.clear();
    }
    if (error_on_failure) {
      throw parse_error("Cannot resolve ID.", in.position());
    }
    return {};
  }

  template <typename I>
  void defineType(const I& in, const SourceId& name, types::ValueType type) {
    if (parse_scope.type_defs.count(name)) {
      throw parse_error("Cannot re-define type", in.position());
    }
    parse_scope.type_defs.emplace(name, type);
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

template <intptr_t kPrefix, Sign sign, typename ActionInput>
static void parseInt(const ActionInput& in, ParseState* state) {
  assert(in.size() > kPrefix);
  const char* num_start = in.begin() + kPrefix;
  intptr_t bits;
  DEBUG_ONLY(auto result =) std::from_chars(num_start, in.end(), bits);
  assert(result.ptr == in.end());

  types::Int data{.width = Bits(bits), .sign = sign};
  validate(data, in.position());

  __ type = types::IntType::get(&__ ctx.ctx_, data);
}

struct UIntType : seq<string<'u', 'i', 'n', 't'>, num> {};

BEGIN_ACTION(UIntType) { parseInt<4, Sign::kUnsigned>(in, state); }
END_ACTION()

struct IntType : seq<string<'i', 'n', 't'>, num> {};

BEGIN_ACTION(IntType) { parseInt<3, Sign::kSigned>(in, state); }
END_ACTION()

struct CharType : seq<string<'c', 'h', 'a', 'r'>, num> {};

BEGIN_ACTION(CharType) { parseInt<4, Sign::kSignless>(in, state); }
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
  assert(!__ type);
  auto id = __ popScopedId(false);
  __ type = __ resolveType(in, id);
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
  assert(__ type);

  types::Array data{.elem = __ type.cast<types::ValueType>(),
                    .length = __ array_len};
  validate(data, in.position());

  __ type = types::ArrayType::get(&__ ctx.ctx_, data);
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
  assert(__ type);

  types::Vector data{
      .elem = __ type,
      .min_length = (__ array_min_len == kNone) ? 0 : __ array_min_len,
      .max_length = __ array_max_len};
  validate(data, in.position());

  __ type = types::VectorType::get(&__ ctx.ctx_, data);

  __ array_max_len = kNone;
  __ array_min_len = kNone;
}
END_ACTION()

struct ArrayModifier : sor<FixedArrayModifier, VarArrayModifier> {};

struct Type : seq<NonArrayType, star<ArrayModifier>> {};

struct FieldDecl : if_must<Id, tok<':'>, Type, tok<';'>> {};

BEGIN_ACTION(FieldDecl) {
  auto field_name = __ popId();
  assert(__ type);
  __ fields[field_name] = __ type;
  __ type = nullptr;
  __ field_order.push_back(field_name);
}
END_ACTION()

struct TopDecl;

struct TypeDecl : if_must<KEYWORD("type"), Id, tok<'='>, Type, tok<';'>> {};

BEGIN_ACTION(TypeDecl) {
  assert(__ type);
  auto name = __ popScopedId();
  __ defineType(in, name, __ type);

  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kType,
      .name = name,
      .type = __ type,
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
  std::vector<types::StructField> fields;
  for (const std::string& name : __ field_order) {
    auto it = __ fields.find(name);
    assert(it != __ fields.end());

    auto type = it->second;
    assert(type);

    fields.push_back(types::StructField{
        .type = type, .name = llvm::StringRef{name.c_str(), name.length()}});
  }

  auto name = __ popScopedId();
  SpanConverter<llvm::StringRef> name_converter{name};
  auto type = types::StructType::get(&__ ctx.ctx_, types::TypeDomain::kHost,
                                     name_converter.get());

  types::Struct data{.fields =
                         Span<types::StructField>{&fields[0], fields.size()}};
  validate(data, in.position());
  type.setTypeData(data);

  __ defineType(in, name, type);

  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kComposite,
      .name = {},
      .type = type,
      .is_external = __ is_external,
  });

  __ fields.clear();
  __ field_order.clear();
  __ is_external = false;
}
END_ACTION()

struct ExplicitTag : sor<num, seq<string<'\''>, any, string<'\''>>> {};

BEGIN_ACTION(ExplicitTag) {
  if (*in.begin() == '\'') {
    __ explicit_tag = in.begin()[1];
  } else {
    auto result = std::from_chars(in.begin(), in.end(), __ explicit_tag);
    if (result.ptr != in.end()) {
      throw parse_error("Invalid tag value '" +
                            std::string(in.begin(), in.end()) + "' provided.\n",
                        in.position());
    }
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
  auto field_name = __ popId();
  // The type may be null, indicating no payload is attached.
  __ fields[field_name] = __ type;
  __ type = nullptr;
  __ field_order.push_back(field_name);
  if (__ explicit_tag) {
    __ explicit_tags[field_name] = __ explicit_tag;
    __ explicit_tag = 0;
  }
}
END_ACTION()

template <typename ActionInput>
static void handleVariant(const ActionInput& in, ParseState* state,
                          bool is_enum) {
  std::vector<types::Term> terms;

  // Set the tags for all the terms.
  std::unordered_set<uint64_t> reserved_tags;
  for (const auto& [_, tag] : __ explicit_tags) {
    reserved_tags.insert(tag);
  }

  uint64_t cur_tag = 1;
  for (const auto& name : __ field_order) {
    auto it = __ fields.find(name);
    assert(it != __ fields.end());

    uint64_t tag;
    if (auto tag_it = __ explicit_tags.find(name);
        tag_it != __ explicit_tags.end()) {
      tag = tag_it->second;
      cur_tag = tag_it->second + 1;
    } else {
      while (reserved_tags.find(cur_tag) != reserved_tags.end()) {
        cur_tag++;
      }
      tag = cur_tag;
      reserved_tags.insert(tag);
    }
    terms.push_back(types::Term{
        .name = llvm::StringRef{name.c_str(), name.length()},
        .type = it->second,
        .tag = tag,
    });
  }

  auto name = __ popScopedId();
  SpanConverter<llvm::StringRef> name_converter{name};
  auto type = types::InlineVariantType::get(
      &__ ctx.ctx_, types::TypeDomain::kHost, name_converter.get());

  types::InlineVariant data{.terms =
                                Span<types::Term>{&terms[0], terms.size()}};
  validate(data, in.position());
  type.setTypeData(data);

  __ defineType(in, name, type);
  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kComposite,
      .name = {},
      .type = type,
      .is_enum = is_enum,
      .is_external = __ is_external,
  });

  __ fields.clear();
  __ field_order.clear();
  __ explicit_tags.clear();
  __ is_external = false;
}

struct VariantDecl : if_must<KEYWORD("variant"), Id, opt<ExternalDecl>, LB,
                             star<VariantFieldDecl>, RB> {};

BEGIN_ACTION(VariantDecl) { handleVariant(in, state, /*is_enum=*/false); }
END_ACTION()

struct EnumFieldDecl : if_must<Id, ExplicitTagDecl, tok<';'>> {};

BEGIN_ACTION(EnumFieldDecl) {
  auto field_name = __ popId();
  __ fields[field_name] = nullptr;
  __ field_order.push_back(field_name);
  if (__ explicit_tag) {
    __ explicit_tags[field_name] = __ explicit_tag;
    __ explicit_tag = 0;
  }
}
END_ACTION()

struct EnumDecl : if_must<KEYWORD("enum"), Id, opt<ExternalDecl>, LB,
                          star<EnumFieldDecl>, RB> {};

BEGIN_ACTION(EnumDecl) { handleVariant(in, state, /*is_enum=*/true); }
END_ACTION()

struct ScopeBegin : LB {};
BEGIN_ACTION(ScopeBegin) {
  auto name = __ popId();
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
  auto id = __ popScopedId();

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
    parseProtoFile(__ parse_scope, found_path);
    __ imports.emplace_back(found_path);
  } else {
    throw parse_error("Cannot find import", in.position());
  }
}
END_ACTION()

struct TagPath : seq<star<seq<identifier, tok<'.'>>>, tok<'_'>> {};

BEGIN_ACTION(TagPath) {
  assert(__ tag_path.empty());
  __ tag_path = types::PathAttr::fromString(
      &__ ctx.ctx_,
      {in.begin(), static_cast<size_t>(std::distance(in.begin(), in.end()))});
}
END_ACTION()

struct TagPathDecl : opt<if_must<tok<'@'>, TagPath>> {};

struct ProtoDecl
    : if_must<KEYWORD("protocol"), Id, tok<':'>, Id, TagPathDecl, tok<';'>> {};

BEGIN_ACTION(ProtoDecl) {
  auto head_name = __ popScopedId();
  auto protocol_name = __ popScopedId();

  if (__ resolveType(in, protocol_name, false)) {
    throw parse_error("Protocol name " + protocol_name.back() +
                          " re-defines existing type name",
                      in.position());
  }

  auto head = __ resolveType(in, head_name);

  // Validate that the tag path points to a variant via struct fields
  if (!__ tag_path.empty()) {
    mlir::Type cur = head;
    const auto& path = __ tag_path.getValue();
    for (uintptr_t i = 0; i < path.size() - 1; ++i) {
      const std::string& term = path[i].str();

      if (!cur.isa<types::StructType>()) {
        throw parse_error("Protocol tag path requests field '" + term +
                              "' in non-struct type",
                          in.position());
        break;
      }

      auto struct_type = cur.cast<types::StructType>();
      auto it =
          std::find_if(struct_type->fields.begin(), struct_type->fields.end(),
                       [&term](const types::StructField& field) {
                         return field.name == term;
                       });
      if (it == struct_type->fields.end()) {
        throw parse_error(
            "Unrecognized term '" + term + "' in protocol tag path",
            in.position());
      }

      cur = it->type;
    }

    if (!cur.isa<types::VariantType>()) {
      throw parse_error("Protocol tag path does not point to a variant type",
                        in.position());
    }
  }

  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kProtocol,
      .name = protocol_name,
      .type = head,
      .tag_path = __ tag_path,
  });

  __ tag_path = types::PathAttr::none(&__ ctx.ctx_);
}
END_ACTION()

struct TopDecl : sor<ImportDecl, StructDecl, VariantDecl, EnumDecl, ProtoDecl,
                     TypeDecl, SpaceDecl> {};

struct ParseFile : must<star<TopDecl>, eof> {};

void parseProtoFile(ParsingScope& scope, const std::filesystem::path& path) {
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
      .ctx = scope.ctx,
      .imports = parsed.imports,
      .decls = parsed.decls,
      .tag_path = types::PathAttr::none(&scope.ctx.ctx_),
  };
  file_input in(path);
  parse<ParseFile, ParseAction>(in, &state);

  scope.stack.pop_back();
  scope.pending_files.erase(path);
}

}  // namespace pj
