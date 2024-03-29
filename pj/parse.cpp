#include <mlir/IR/MLIRContext.h>
#include <charconv>
#include <pegtl.hpp>
#include <pegtl/contrib/analyze.hpp>

#include "arch.hpp"
#include "array_ref.hpp"
#include "protogen.hpp"
#include "types.hpp"
#include "validate.hpp"

#include <unordered_set>

using namespace tao::pegtl;

namespace pj {

template <typename Path>
std::string getPathAsString(const Path& path) {
  std::string str;
  bool first = true;
  for (auto& piece : path) {
    if (!first) {
      str += ".";
    }
    first = false;
    str += piece;
  }
  return str;
}

struct ParseState {
  ParsingScope& parse_scope;
  mlir::MLIRContext& ctx;
  const bool parse_as_spec;

  std::vector<std::filesystem::path>& imports;
  std::vector<ParsedProtoFile::Decl>& decls;
  std::vector<std::pair<SourceId, ParsedProtoFile::Protocol>>& protos;
  std::map<SourceId, ParsedProtoFile::Precomp, SourceIdLess>& precomps;
  std::set<SourceId, SourceIdLess>& portals;

  // Updated after parsing a 'FieldDecl' rule.
  // Cleared when done with a struct/variant decl.
  std::map<std::string, pj::types::ValueType> fields;

  // Populated after parsing a `VariantFieldDecl` or `EnumFieldDecl` rule.
  // Cleared after parsing a `VariantDecl` or `EnumDecl` rule.
  std::map<std::string, uint64_t> explicit_tags;

  // Populated after parsing a DefaultDecl.
  // Cleared after parsing VariantFieldDecl or EnumFieldDecl.
  bool is_default;

  // Populated after parsing a VariantFieldDecl or EnumFieldDecl.
  // Cleared after parsing VariantDecl or EnumDecl.
  std::string default_term;

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

  // Populated after parsing PathDecl.
  // Cleared after parsing ProtoDecl, SizerDecl, EncoderDecl, or DecoderDecl.
  // The meaning is interpreted as tag_path for ProtoDecl, as src_path for
  // SizerDecl and EncoderDecl, and as handlers for DecoderDecl.
  std::vector<types::PathAttr> paths;

  SourceId space;

  // Populated after parsing Id and ScopedId.
  // Cleared by popId() and popScopedId().
  std::vector<std::vector<std::string>> ids = {{}};

  // Populated after parsing RoundUpDecl.
  // Cleared by SizerDecl.
  bool round_up = false;

  // Populated after parsing SizerDecl, EncoderDecl, and DecoderDecl,
  // respectively.
  // Cleared by PortalDecl.
  std::vector<Portal::Sizer> sizers;
  std::vector<Portal::Encoder> encoders;
  std::vector<Portal::Decoder> decoders;

  // Populated after parsing ProtoParam. Cleared by PrecompDecl and PortalDecl.
  std::optional<SourceId> proto_param;

  std::string language_text;

  // Returns PathAttr::none if paths is empty, paths[0] if it's not empty, and
  // asserts if there is more than one entry.
  types::PathAttr getSinglePath() {
    assert(paths.size() <= 1);
    return paths.empty() ? types::PathAttr::none(&ctx) : paths[0];
  }

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
  std::pair<SourceId, types::ValueType> resolveType(
      const I& in, const SourceId& id, bool error_on_failure = true) {
    auto result = resolve(in, id, parse_scope.type_defs, error_on_failure);
    if (result) {
      return *result;
    }
    return {};
  }

  template <typename I>
  std::optional<
      std::pair<SourceId, std::pair<types::ValueType, types::PathAttr>>>
  resolveSpec(const I& in, const SourceId& id, bool error_on_failure = true) {
    return resolve(in, id, parse_scope.spec_defs, error_on_failure);
  }

  template <typename I, typename M>
  auto resolve(const I& in, const SourceId& id, const M& map,
               bool error_on_failure = true)
      -> std::optional<typename M::value_type> {
    std::vector<llvm::StringRef> id_vec;
    id_vec.reserve(space.size() + id_vec.size());
    for (intptr_t i = space.size(); i >= 0; --i) {
      for (intptr_t j = 0; j < i; ++j) {
        id_vec.push_back(space[j]);
      }
      for (auto& p : id) {
        id_vec.push_back(p);
      }

      if (auto it = map.find(id_vec); it != map.end()) {
        return *it;
      }
      id_vec.clear();
    }
    if (error_on_failure) {
      throw parse_error("Cannot resolve type " + getPathAsString(id),
                        in.position());
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

template <intptr_t kPrefix, Sign sign, typename ActionInput>
static void parseInt(const ActionInput& in, ParseState* state) {
  assert(in.size() > kPrefix);
  const char* num_start = in.begin() + kPrefix;
  intptr_t bits;
  DEBUG_ONLY(auto result =) std::from_chars(num_start, in.end(), bits);
  assert(result.ptr == in.end());

  types::Int data{
      .width = Bits(bits),
      .alignment = Bits(bits),
      .sign = sign,
  };
  validate(data, in.position());

  __ type = types::IntType::get(&__ ctx, data);
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

struct FloatType : seq<string<'f', 'l', 'o', 'a', 't'>, num> {};

BEGIN_ACTION(FloatType) {
  assert(in.size() > 5);
  const char* num_start = in.begin() + 5;
  intptr_t bits;
  std::from_chars(num_start, in.end(), bits);

  if (bits != types::Float::k32 && bits != types::Float::k64) {
    throw parse_error("Only 32-bit and 64-bit floats are supported.",
                      in.position());
  }

  types::Float data{
      .width = static_cast<types::Float::FloatWidth>(bits),
      .alignment = Bits(bits),
  };

  __ type = types::FloatType::get(&__ ctx, data);
}
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
  __ type = __ resolveType(in, id).second;
}
END_ACTION()

struct AnyType : KEYWORD("any") {};

BEGIN_ACTION(AnyType) {
  assert(!__ type);
  __ type = types::AnyType::get(&__ ctx, types::Any{});
}
END_ACTION()

struct NonArrayType
    : sor<UIntType, IntType, CharType, FloatType, AnyType, TypeRef> {};

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

  __ type = types::ArrayType::get(&__ ctx, data);
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
      .min_length =
          __ array_min_len < 0 ? 0 : static_cast<uint64_t>(__ array_min_len),
      .max_length = __ array_max_len,
      .elem_width = Bytes(0)};
  validate(data, in.position());

  __ type = types::VectorType::get(&__ ctx, data);

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

struct ExternalDecl : KEYWORD("external") {};

BEGIN_ACTION(ExternalDecl) { __ is_external = true; }
END_ACTION()

struct TypeDecl : if_must<KEYWORD("type"), Id, opt<ExternalDecl>, tok<'='>,
                          Type, tok<';'>> {};

BEGIN_ACTION(TypeDecl) {
  assert(__ type);
  auto name = __ popScopedId();
  auto type = __ type;
  __ type = nullptr;

  if (__ is_external) {
    if (!__ parse_as_spec) {
      // Don't define this type as the type written in the .pj source;
      // rather, define it as an empty struct with the given name. Also
      // don't add it as a declaration; it is up to the user provide one.
      ArrayRefConverter<llvm::StringRef> name_converter{name};
      type = types::StructType::get(&__ ctx,
                                    types::InternalDomainAttr::get(&__ ctx),
                                    name_converter.get());
    }
  } else {
    __ decls.emplace_back(ParsedProtoFile::Decl{
        .kind = ParsedProtoFile::DeclKind::kType,
        .name = name,
        .type = type,
    });
  }

  __ defineType(in, name, type);

  __ is_external = false;
}
END_ACTION();

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
  ArrayRefConverter<llvm::StringRef> name_converter{name};
  auto type = types::StructType::get(
      &__ ctx, types::InternalDomainAttr::get(&__ ctx), name_converter.get());

  types::Struct data{
      .fields = ArrayRef<types::StructField>{fields.data(), fields.size()}};
  validate(data, in.position());
  type.setTypeData(data);

  __ defineType(in, name, type);

  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kComposite,
      .name = name,
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

struct DefaultDecl : KEYWORD("default") {};

BEGIN_ACTION(DefaultDecl) { __ is_default = true; }
END_ACTION()

struct ExplicitTagDecl : opt<if_must<tok<'='>, ExplicitTag>> {};

template <typename ActionInput>
static void handleVariantField(const ActionInput& in, ParseState* state,
                               bool is_enum) {
  auto field_name = __ popId();
  if (is_enum) {
    __ fields[field_name] = types::UnitType::get(&__ ctx);
  } else {
    // The type may be null, indicating no payload is attached.
    __ fields[field_name] = __ type ? __ type : types::UnitType::get(&__ ctx);
    __ type = nullptr;
  }
  __ field_order.push_back(field_name);
  if (__ explicit_tag) {
    __ explicit_tags[field_name] = __ explicit_tag;
    __ explicit_tag = 0;
  }
  if (__ is_default) {
    if (!__ default_term.empty()) {
      throw parse_error("Cannot have multiple default terms.\n", in.position());
    }
    __ default_term = field_name;
    __ is_default = false;
  }
}

struct VariantFieldDecl : if_must<Id, opt<tok<':'>, Type>, ExplicitTagDecl,
                                  opt<DefaultDecl>, tok<';'>> {};

BEGIN_ACTION(VariantFieldDecl) { handleVariantField(in, state, false); }
END_ACTION()

struct EnumFieldDecl
    : if_must<Id, ExplicitTagDecl, opt<DefaultDecl>, tok<';'>> {};

BEGIN_ACTION(EnumFieldDecl) { handleVariantField(in, state, true); }
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

  // Add undef default term if there is no default term, and move the default
  // term to the beginning if there is.
  size_t default_term_idx;
  if (__ default_term.empty()) {
    __ fields["undef"] = types::UnitType::get(&__ ctx);
    __ field_order.insert(__ field_order.begin(), "undef");
    default_term_idx = 0;
  } else {
    auto it = std::find(__ field_order.begin(), __ field_order.end(),
                        __ default_term);
    assert(it != __ field_order.end());
    default_term_idx = std::distance(__ field_order.begin(), it);
  }

  // Set tag values that were not explicitly provided.
  uint64_t cur_tag = 0;
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
  ArrayRefConverter<llvm::StringRef> name_converter{name};
  auto type = types::InlineVariantType::get(
      &__ ctx, types::InternalDomainAttr::get(&__ ctx), name_converter.get());

  types::InlineVariant data{
      .terms = ArrayRef<types::Term>{terms.data(), terms.size()},
      .default_term = default_term_idx,
  };
  validate(data, in.position());
  type.setTypeData(data);

  __ defineType(in, name, type);
  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kComposite,
      .name = name,
      .type = type,
      .is_enum = is_enum,
      .is_external = __ is_external,
  });

  __ fields.clear();
  __ field_order.clear();
  __ explicit_tags.clear();
  __ is_external = false;
  __ default_term.clear();
}

struct VariantDecl : if_must<KEYWORD("variant"), Id, opt<ExternalDecl>, LB,
                             star<VariantFieldDecl>, RB> {};

BEGIN_ACTION(VariantDecl) { handleVariant(in, state, /*is_enum=*/false); }
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
    parseProtoFile(__ parse_scope, found_path, __ parse_as_spec);
    __ imports.emplace_back(found_path);
  } else {
    throw parse_error("Cannot find import", in.position());
  }
}
END_ACTION()

struct Path : seq<star<seq<identifier, tok<'.'>>>, identifier> {};

BEGIN_ACTION(Path) {
  __ paths.push_back(types::PathAttr::fromString(
      &__ ctx,
      {in.begin(), static_cast<size_t>(std::distance(in.begin(), in.end()))}));
}
END_ACTION()

struct PathDecl : opt<if_must<tok<'@'>, Path>> {};

// Validate that path_attr points to a variant via struct fields starting from
// head. If check_term is set, additionally validates that the final piece in
// path_attr actually corresponds to a term in the variant.
template <typename ActionInput>
void checkVariantPath(const ActionInput& in, types::ValueType head,
                      types::PathAttr path_attr, bool check_term) {
  const auto& path = path_attr.getValue();
  auto path_str = getPathAsString(path);
  auto cur = head;
  for (uintptr_t i = 0; i < path.size() - 1; ++i) {
    const std::string& term = path[i].str();

    if (!cur.isa<types::StructType>()) {
      throw parse_error("Path " + path_str + " requests field '" + term +
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
      throw parse_error("Unrecognized term '" + term + "' in path " + path_str,
                        in.position());
    }

    cur = it->type;
  }

  if (auto var = cur.dyn_cast<types::InlineVariantType>()) {
    if (check_term) {
      auto it = std::find_if(
          var->terms.begin(), var->terms.end(),
          [&](const types::Term& t) { return t.name == path.back(); });
      if (it == var->terms.end()) {
        throw parse_error("Path " + getPathAsString(path) +
                              " does not match any term in the variant",
                          in.position());
      }
    }
    return;
  }

  throw parse_error("Path does not point to a variant type", in.position());
}

struct SpecDecl : if_must<KEYWORD("specification"), Id, tok<':'>, ScopedId,
                          PathDecl, tok<';'>> {};

BEGIN_ACTION(SpecDecl) {
  auto head_name = __ popScopedId(false);
  auto spec_name = __ popScopedId();

  if (__ resolveSpec(in, spec_name, /*error_on_failure=*/false)) {
    throw parse_error("Specification name " + spec_name.back() +
                          " re-defines existing specification",
                      in.position());
  }

  auto head = __ resolveType(in, head_name).second;

  auto tag_path = __ getSinglePath();
  if (!tag_path.empty()) {
    const auto& path = tag_path.getValue();
    if (path.back() != "_") {
      throw parse_error("Tag path must end with _ following variant term",
                        in.position());
    }
    checkVariantPath(in, head, tag_path, /*check_term=*/false);
  }

  __ protos.emplace_back(spec_name, std::make_pair(head, tag_path));
  __ parse_scope.spec_defs.emplace(spec_name, std::make_pair(head, tag_path));

  __ paths.clear();
}
END_ACTION()

struct RoundUpDecl : KEYWORD("round_up") {};

BEGIN_ACTION(RoundUpDecl) { __ round_up = true; }
END_ACTION()

struct SizerDecl;
struct EncoderDecl;

template <typename Decl, typename ActionInput>
void handleSizerOrEncoderDecl(const ActionInput& in, ParseState* state) {
  static_assert(std::is_same_v<Decl, SizerDecl> ||
                std::is_same_v<Decl, EncoderDecl>);

  auto src = __ popScopedId(false);
  // Check that the src type exists.
  auto [src_name, src_type] = __ resolveType(in, src);

  auto name = __ popId();

  auto src_path = __ getSinglePath();
  if (!src_path.empty()) {
    checkVariantPath(in, src_type, src_path, /*check_term=*/true);
  }

  if constexpr (std::is_same_v<Decl, SizerDecl>) {
    __ sizers.push_back({
        .name = name,
        .src = src_name,
        .src_path = src_path,
        .round_up = __ round_up,
    });
  } else {
    __ encoders.push_back({
        .name = name,
        .src = src_name,
        .src_path = src_path,
    });
  }

  __ paths.clear();
  if constexpr (std::is_same_v<Decl, SizerDecl>) {
    __ round_up = false;
  }
}

struct SizerDecl : if_must<KEYWORD("sizer"), Id, opt<RoundUpDecl>, tok<':'>,
                           ScopedId, opt<PathDecl>, tok<';'>> {};

BEGIN_ACTION(SizerDecl) { handleSizerOrEncoderDecl<SizerDecl>(in, state); }
END_ACTION()

struct EncoderDecl : if_must<KEYWORD("encoder"), Id, tok<':'>, ScopedId,
                             opt<PathDecl>, tok<';'>> {};

BEGIN_ACTION(EncoderDecl) { handleSizerOrEncoderDecl<EncoderDecl>(in, state); }
END_ACTION()

struct HandlersDecl : if_must<LB, KEYWORD("handlers"), tok<'['>, Path,
                              star<seq<tok<','>, Path>>, opt<tok<','>>,
                              tok<']'>, tok<';'>, RB> {};

struct DecoderDecl : if_must<KEYWORD("decoder"), Id, tok<':'>, ScopedId,
                             opt<HandlersDecl>, tok<';'>> {};

BEGIN_ACTION(DecoderDecl) {
  auto dst = __ popScopedId(false);
  // Check that the dst type exists.
  auto [dst_name, dst_type] = __ resolveType(in, dst);

  for (auto handler : __ paths) {
    checkVariantPath(in, dst_type, handler, /*check_term=*/true);
  }

  // Check for duplicates
  for (auto it = __ paths.begin(); it != __ paths.end(); ++it) {
    if (std::find(std::next(it), __ paths.end(), *it) != __ paths.end()) {
      throw parse_error(
          "Duplicate handler term " + getPathAsString(it->getValue()),
          in.position());
    }
  }

  __ decoders.push_back({
      .name = __ popId(),
      .dst = dst_name,
      .handlers = __ paths,
  });

  __ paths.clear();
}
END_ACTION()

struct ProtoParam : ScopedId {};
BEGIN_ACTION(ProtoParam) {
  assert(!__ proto_param.has_value());
  auto id = __ popScopedId(/*include_space=*/false);
  __ proto_param = __ resolveSpec(in, id)->first;
}
END_ACTION()

struct PrecompDecl : if_must<KEYWORD("precompile"), Id, tok<':'>, ScopedId,
                             tok<'('>, ProtoParam, tok<')'>, tok<';'>> {};
BEGIN_ACTION(PrecompDecl) {
  ASSERT(__ proto_param.has_value());
  auto portal_name = __ popScopedId(/*include_space=*/false);
  auto precomp_name = __ popScopedId();

  auto resolved_portal = __ resolve(in, portal_name, __ portals);
  if (!resolved_portal) {
    throw parse_error("No portal named " + getPathAsString(portal_name),
                      in.position());
  }

  // Make sure there isn't also another class of this name.
  if (__ resolve(in, precomp_name, __ precomps, /*error_on_failure=*/false) ||
      __ resolve(in, precomp_name, __ portals, /*error_on_failure=*/false)) {
    throw parse_error("Redefining class " + getPathAsString(precomp_name),
                      in.position());
  }

  __ precomps.emplace(precomp_name,
                      std::make_pair(*resolved_portal, *__ proto_param));
  __ proto_param = {};
}
END_ACTION()

struct PortalDecl
    : if_must<KEYWORD("portal"), Id, tok<':'>, ProtoParam, LB,
              star<sor<SizerDecl, EncoderDecl, DecoderDecl>>, RB> {};

BEGIN_ACTION(PortalDecl) {
  auto name = __ popScopedId();

  if (__ portals.count(name)) {
    throw parse_error("Multiple portals", in.position());
  }

  auto& portal = __ parse_scope.portal_defs
                     .emplace(name,
                              Portal{
                                  .sizers = __ sizers,
                                  .encoders = __ encoders,
                                  .decoders = __ decoders,
                              })
                     .first->second;
  __ portals.emplace(name);

  portal.proto = *__ proto_param;
  __ proto_param = {};

  __ sizers.clear();
  __ encoders.clear();
  __ decoders.clear();
}
END_ACTION()

struct LanguageLiteralDelimiter : string<'"', '"', '"'> {};

struct LanguageText : any {};

BEGIN_ACTION(LanguageText) { __ language_text += in.string(); }
END_ACTION()

struct LanguageDecl : if_must<KEYWORD("language"), Id, space_or_comment,
                              LanguageLiteralDelimiter,
                              until<LanguageLiteralDelimiter, LanguageText>> {};

BEGIN_ACTION(LanguageDecl) {
  __ decls.emplace_back(ParsedProtoFile::Decl{
      .kind = ParsedProtoFile::DeclKind::kLanguage,
      .language_space = __ space,
      .language = __ popId(),
      .language_text = __ language_text,
  });
  __ language_text.clear();
}
END_ACTION()

struct TopDecl
    : sor<ImportDecl, StructDecl, VariantDecl, EnumDecl, SpecDecl, TypeDecl,
          SpaceDecl, LanguageDecl, PortalDecl, PrecompDecl> {};

struct ParseFile : must<star<TopDecl>, eof> {};

void parseProtoFile(ParsingScope& scope, const std::filesystem::path& path,
                    bool parse_as_spec) {
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
      .parse_as_spec = parse_as_spec,
      .imports = parsed.imports,
      .decls = parsed.decls,
      .protos = parsed.spec_defs,
      .precomps = parsed.precomps,
      .portals = parsed.portals,
  };
  file_input in(path);
  parse<ParseFile, ParseAction>(in, &state);

  scope.stack.pop_back();
  scope.pending_files.erase(path);
}

}  // namespace pj
