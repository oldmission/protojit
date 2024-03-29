#include <algorithm>
#include <charconv>
#include <cmath>
#include <unordered_map>

#include "defer.hpp"
#include "protogen.hpp"
#include "sourcegen.hpp"
#include "types.hpp"

namespace pj {

std::ostream& operator<<(std::ostream& os, llvm::StringRef str) {
  return os << std::string_view(str);
}

std::string convertSign(Sign sign) {
  switch (sign) {
    case Sign::kSigned:
      return "PJ_SIGN_SIGNED";
    case Sign::kUnsigned:
      return "PJ_SIGN_UNSIGNED";
    case Sign::kSignless:
      return "PJ_SIGN_SIGNLESS";
    default:
      UNREACHABLE();
  }
}

void SourceGenerator::printIntTypeRef(Width width, Sign sign, bool wrap,
                                      std::string decl) {
  assert(width.bytes() > 0);

  auto& os = stream();
  if (wrap && !decl.empty()) {
    os << "::pj::integer_validate<decltype(" << decl << "), ";
  }

  if (wrap) os << "::pj::integer<";

  if (sign == Sign::kSignless) {
    // TODO: maybe use wchar_t?
  }
  if (!wrap && sign == Sign::kUnsigned) os << "unsigned ";
  if (!wrap && sign == Sign::kSigned) os << "signed ";

  auto log = static_cast<uintptr_t>(std::ceil(std::log2(width.bytes())));

  // TODO: platform-specific
  std::array<std::string_view, 4> kIntTypes{"char", "short", "int", "long"};
  assert(log <= kIntTypes.size());
  if (wrap) {
    os << width.bits() << " ";
  } else {
    os << kIntTypes[log];
  }

  if (wrap) os << ", " << convertSign(sign) << ">";

  if (wrap && !decl.empty()) {
    os << ">::type";
  }
}

llvm::StringRef SourceGenerator::classNameFor(types::ValueType type) {
  if (auto named = type.dyn_cast<types::NominalType>()) {
    return named.name().back();
  }

  if (type.isa<types::AnyType>()) {
    return "Any";
  }

  if (type.isa<types::ArrayType>()) {
    return "array";
  }

  if (type.isa<types::UnitType>()) {
    return "Unit";
  }

  if (type.isa<types::VectorType>()) {
    return "span";
  }

  return {};
}

void SourceGenerator::printTypeRef(types::ValueType type, bool wrap,
                                   std::string decl) {
  auto& os = stream();

  if (auto named = type.dyn_cast<types::NominalType>()) {
    printName(named.name());
    return;
  }

  if (type.isa<types::AnyType>()) {
    os << "pj::Any";
    base_headers_.emplace("any");
    return;
  }

  if (auto I = type.dyn_cast<types::IntType>()) {
    printIntTypeRef(I->width, I->sign, wrap, decl);
    base_headers_.emplace("integer");
    return;
  }

  if (auto F = type.dyn_cast<types::FloatType>()) {
    os << (F->width == types::Float::k32 ? "float" : "double");
    base_headers_.emplace("float");
    return;
  }

  if (auto U = type.dyn_cast<types::UnitType>()) {
    os << "pj::Unit";
    base_headers_.emplace("unit");
    return;
  }

  if (auto A = type.dyn_cast<types::ArrayType>()) {
    os << "pj::array<";
    printTypeRef(A->elem, true);
    os << ", " << A->length << ">";
    base_headers_.emplace("array");
    return;
  }

  if (auto V = type.dyn_cast<types::VectorType>()) {
    os << "pj::span<";
    printTypeRef(V->elem, true);
    auto max_length = V->max_length == kNone ? "::pj::unbounded_length"
                                             : std::to_string(V->max_length);
    os << ", " << max_length << ", " << V->min_length << ">";
    base_headers_.emplace("span");
    return;
  }

  UNREACHABLE();
}

std::string SourceGenerator::createTypeHandle(std::string decl,
                                              types::ValueType type) {
  ASSERT(region_ == Region::kBuilders);

  std::string handle = getUniqueName();
  auto& os = stream();

  if (domain_ == Domain::kHost || type.isa<types::NominalType>()) {
    os << "const auto* " << handle << " = BuildPJType<";
    if (domain_ == Domain::kWire || decl.empty() ||
        type.isa<types::IntType>() || type.isa<types::FloatType>() ||
        type.isa<types::UnitType>()) {
      printTypeRef(type, /*wrap=*/true, decl);
    } else {
      os << "decltype(" << decl << ")";
    }
    os << ">::build(ctx, domain);\n";
    return handle;
  }

  // Copy the type information exactly as-is.
  if (auto I = type.dyn_cast<types::IntType>()) {
    os << "const auto* " << handle << " = PJCreateIntType(ctx";
    os << ", /*width=*/" << I->width.bits();
    os << ", /*alignment=*/" << I->alignment.bits();
    os << ", /*sign=*/" << convertSign(I->sign) << ");\n";
    return handle;
  }

  if (auto F = type.dyn_cast<types::FloatType>()) {
    os << "const auto* " << handle << " = PJCreateFloatType(ctx";
    os << ", /*width=*/"
       << (F->width == types::Float::k32 ? "PJ_FLOAT_WIDTH_32"
                                         : "PJ_FLOAT_WIDTH_64");
    os << ", /*alignment=*/" << F->alignment.bits();
    return handle;
  }

  if (auto U = type.dyn_cast<types::UnitType>()) {
    os << "const PJUnitType* " << handle << " = PJCreateUnitType(ctx);\n";
    return handle;
  }

  if (auto A = type.dyn_cast<types::ArrayType>()) {
    std::string elem_handle = createTypeHandle("", A->elem);
    os << "const auto* " << handle << " = PJCreateArrayType(ctx";
    os << ", /*elem=*/" << elem_handle;
    os << ", /*length=*/" << A->length;
    os << ", /*elem_size=*/" << A->elem_size.bits();
    os << ", /*alignment=*/" << A->alignment.bits() << ");\n";
    return handle;
  }

  if (auto V = type.dyn_cast<types::VectorType>()) {
    std::string elem_handle = createTypeHandle("", V->elem);
    os << "const auto* " << handle << " = PJCreateVectorType(ctx";
    os << ", /*elem=*/" << elem_handle;
    os << ", /*min_length=*/" << V->min_length;
    os << ", /*max_length=*/" << V->max_length;
    os << ", /*wire_min_length=*/" << V->min_length;
    os << ", /*ppl_count=*/" << V->ppl_count;
    os << ", /*length_offset=*/" << V->length_offset.bits();
    os << ", /*length_size=*/" << V->length_size.bits();
    os << ", /*ref_offset=*/" << V->ref_offset.bits();
    os << ", /*ref_size=*/" << V->ref_size.bits();
    os << ", /*reference_mode=*/";
    switch (V->reference_mode) {
      case ReferenceMode::kOffset:
        os << "PJ_REFERENCE_MODE_OFFSET";
        break;
      case ReferenceMode::kPointer:
        os << "PJ_REFERENCE_MODE_POINTER";
        break;
      default:
        UNREACHABLE();
    }
    os << ", /*inline_payload_offset=*/" << V->inline_payload_offset.bits();
    os << ", /*inline_payload_size=*/" << V->inline_payload_size.bits();
    os << ", /*partial_payload_offset=*/" << V->partial_payload_offset.bits();
    os << ", /*partial_payload_size=*/" << V->partial_payload_size.bits();
    os << ", /*size=*/" << V->size.bits();
    os << ", /*alignment=*/" << V->alignment.bits();
    os << ", /*outlined_payload_alignment=*/"
       << V->outlined_payload_alignment.bits() << ");\n";
    return handle;
  }

  UNREACHABLE();
}

void SourceGenerator::addProtocol(const SourceId& name,
                                  ParsedProtoFile::Protocol proto) {
  if (protos_.count(name)) return;
  cpp_ << "auto " << getNameAsString(name, "_") << " __attribute__((unused)) = "
       << "ctx_.plan<"
       << getNameAsString(proto.first.cast<types::NominalType>().name())
       << ">(\"" << proto.second << "\");\n";
  protos_.emplace(name);
}

std::string SourceGenerator::buildStringArray(ArrayRef<llvm::StringRef> arr) {
  auto& os = stream();
  std::string var;
  os << "const char* " << (var = getUniqueName()) << "[" << arr.size()
     << "] = {";
  for (uintptr_t i = 0; i < arr.size(); ++i) {
    os << "\"" << std::string_view(arr[i]) << "\"";
    if (i < arr.size() - 1) {
      os << ", ";
    }
  }
  os << "};\n";
  return var;
}

void SourceGenerator::addTypedef(const SourceId& name, types::ValueType type) {
  region_ = Region::kDefs;
  auto& os = stream();

  beginNamespaceOf(name);
  os << "using " << std::string_view(name.back()) << " = ";
  printTypeRef(type);
  os << ";\n";
  endNamespaceOf(name);
}

void SourceGenerator::addType(types::ValueType type) {
  if (auto nominal = type.dyn_cast<types::NominalType>()) {
    if (nominal.domain().isa<types::WireDomainAttr>() &&
        generated_.find(type.getAsOpaquePointer()) == generated_.end()) {
      addComposite(nominal);
    }
    return;
  }

  if (auto arr = type.dyn_cast<types::ArrayType>()) {
    addType(arr->elem);
    return;
  }

  if (auto vec = type.dyn_cast<types::VectorType>()) {
    addType(vec->elem);
    return;
  }
}

void SourceGenerator::addWireProtocol(const SourceId& name,
                                      types::ProtocolType proto) {
  pushDomain(Domain::kWire);

  addType(proto->head);

  region_ = Region::kDefs;
  beginNamespaceOf(name);
  stream() << "struct " << std::string_view(name.back()) << ";\n";
  endNamespaceOf(name);

  region_ = Region::kBuilders;
  auto& os = stream();
  os << "namespace pj {\n";
  os << "namespace gen {\n";
  os << "template <>\n"
     << "struct BuildPJProtocol<";
  printName(name);
  os << "> {\n"
     << "using Head = ";
  printTypeRef(proto->head);
  os << ";\n";

  os << "static const auto* build(PJContext* ctx, const PJDomain* domain) {\n";
  std::string head_handle = createTypeHandle("Head", proto->head);
  os << "return PJCreateProtocolType(ctx, " << head_handle << ", "
     << proto->buffer_offset.bits() << ");\n";
  os << "}\n";

  os << "};\n}  // namespace gen\n\n";
  os << "\n}  // namespace pj\n\n";

  popDomain();
}

void SourceGenerator::addComposite(types::ValueType type, bool is_external) {
  pushDomain(type.cast<types::NominalType>().domain());

  if (type.isa<types::StructType>()) {
    addStruct(type.cast<types::StructType>(), is_external);
  } else {
    addVariant(type.cast<types::VariantType>(), is_external);
  }

  generated_.insert(type.getAsOpaquePointer());
  popDomain();
}

void SourceGenerator::addStructDef(types::StructType type, bool decl_only) {
  region_ = Region::kDefs;

  auto name = type.cast<types::NominalType>().name();
  beginNamespaceOf(name);

  stream() << "struct " << std::string_view(name.back());
  if (!decl_only) {
    stream() << " {\n";
    for (auto& field : type->fields) {
      printTypeRef(field.type);
      stream() << " " << std::string_view(field.name) << ";\n";
    }
    stream() << "};\n\n";
  } else {
    stream() << ";\n";
  }

  endNamespaceOf(name);
}

void SourceGenerator::addStructBuilder(types::StructType type,
                                       bool is_external) {
  region_ = Region::kBuilders;

  auto& os = stream();
  auto name = type.cast<types::NominalType>().name();

  os << "namespace pj {\n";
  os << "namespace gen {\n";
  os << "template <>\n"
     << "struct BuildPJType<";
  printName(name);
  os << "> {\n";

  os << "static const auto* build(PJContext* ctx, const PJDomain* domain) {\n";

  // Generate an array of handles for each of the fields.
  os << "const PJStructField* fields[" << type->fields.size() << "];\n";
  uintptr_t field_num = 0;
  for (auto& field : type->fields) {
    std::vector<llvm::StringRef> field_name{name.begin(), name.end()};
    field_name.emplace_back(field.name);
    auto field_handle =
        createTypeHandle(getNameAsString(field_name), field.type);

    os << "fields[" << field_num++ << "] = PJCreateStructField(";
    os << "/*name=*/\"" << std::string_view(field.name) << "\"";
    os << ", /*type=*/" << field_handle;
    if (domain_ == Domain::kHost) {
      os << ", /*offset=*/offsetof(";
      printName(name);
      os << ", " << std::string_view(field.name) << ") << 3";
    } else {
      os << ", /*offset=*/" << field.offset.bits();
    }
    os << ");\n";
  }

  // Generate an array containing the name of the struct.
  auto name_array = buildStringArray(name);

  // Generate the final struct handle.
  std::string handle;
  os << "const PJStructType* " << (handle = getUniqueName())
     << " = PJCreateStructType(ctx";
  os << ", /*name_size=*/" << name.size();
  os << ", /*name=*/" << name_array;
  os << ", /*type_domain=*/domain";
  os << ", /*num_fields=*/" << type->fields.size();
  os << ", /*fields=*/fields";
  if (domain_ == Domain::kHost) {
    os << ", /*size=*/sizeof(";
    printName(name);
    os << ") << 3";
    os << ", /*alignment=*/alignof(";
    printName(name);
    os << ") << 3";
  } else {
    os << ", /*size=*/" << type->size.bits();
    os << ", /*alignment=*/" << type->alignment.bits();
  }
  os << ");\n";

  os << "return " << handle << ";\n"
     << "}\n"
     << "};\n"
     << "}  // namespace gen\n"
     << "}  // namespace pj\n"
     << "\n";
}

void SourceGenerator::addStruct(types::StructType type, bool is_external) {
  for (const auto& field : type->fields) {
    addType(field.type);
  }

  if (domain_ == Domain::kHost && !is_external) {
    addStructDef(type, false);
  } else if (domain_ == Domain::kWire) {
    addStructDef(type, true);
  }

  addStructBuilder(type, is_external);
}

void SourceGenerator::addVariantFunctions(types::VariantType type) {
  auto& os = stream();

  const bool default_ctor =
      type.terms()[type.default_term()].type.isa<types::UnitType>();
  if (default_ctor) {
    // Add default constructor with no args that sets undef tag.
    os << type.name().back()
       << "() : tag(Kind::" << type.terms()[type.default_term()].name
       << ") {}\n";
  }

  // Add a destructor. This needs to call the destructor for whichever term is
  // set.
  os << "~" << type.name().back() << "() { deinitialize(); }\n";

  // Add constructors for each non-undef term.
  // Each constructor gets a token type that distinguishes it from the others
  // during function resolution.
  for (auto& term : type.terms()) {
    os << "struct marker_" << term.name << " {};\n"
       << "constexpr static marker_" << term.name << " " << term.name
       << " = marker_" << term.name << "();\n"
       << "template <typename... Args>\n"
       << type.name().back() << "(marker_" << term.name << " " << term.name
       << ", Args&&... args) : tag(Kind::" << term.name << ") {\n"
       << "  new (&value." << term.name << ") ";
    printTypeRef(term.type);
    os << "(std::forward<Args>(args)...);\n"
       << "}";
  }

  // Add a copy constructor.
  os << type.name().back() << "(const " << type.name().back() << "& other) {\n"
     << "  initialize_from(other);\n"
     << "}\n";

  // Add a copy assignment operator. First deinitialize whatever term is
  // currently set, then copy over the tag and copy-construct the associated
  // term.
  os << type.name().back() << "& operator=(const " << type.name().back()
     << "& other) {\n"
     << "deinitialize();\n"
     << "initialize_from(other);\n"
     << "return *this;\n"
     << "}\n";

  os << "private:\n";

  // initialize_from(...) {...}
  os << "void initialize_from(const " << type.name().back() << "& other) {\n"
     << "tag = other.tag;\n"
     << "switch(tag) {\n";

  for (auto& term : type.terms()) {
    os << "case Kind::" << term.name << ":\n"
       << "  new (&value." << term.name << ") ";
    printTypeRef(term.type);
    os << "(other.value." << term.name << ");\n"
       << "break;\n";
  }

  os << "default:;\n"
     << "}\n"
     << "}\n";

  // deinitialize() {...}
  os << "void deinitialize() {\n"
     << "switch (tag) {\n";

  for (auto& term : type.terms()) {
    os << "case Kind::" << term.name << ":\n";
    auto class_name = classNameFor(term.type);
    if (!class_name.empty()) {
      os << "value." << term.name << ".";
      printTypeRef(term.type);
      os << "::~" << class_name << "();\n";
    }
    os << "break;\n";
  }

  os << "default:;\n"
     << "}\n"
     << "}";
}

void SourceGenerator::addVariantDef(types::VariantType type, bool has_value,
                                    Width tag_width, bool decl_only) {
  region_ = Region::kDefs;

  auto& os = stream();
  auto name = type.cast<types::NominalType>().name();

  beginNamespaceOf(name);

  if (has_value) {
    os << "struct " << std::string_view(name.back());
    if (!decl_only) {
      os << " {\n";
      os << "union Union {\n"
         << "  Union() {}\n"
         << "char __;\n";
      for (const auto& term : type.terms()) {
        if (term.type) {
          printTypeRef(term.type);
          os << " " << std::string_view(term.name) << ";\n";
        }
      }
      os << "} value;\n";
      os << "enum class Kind : ";
      printIntTypeRef(tag_width, Sign::kUnsigned);
    }
  } else {
    os << "enum class " << std::string_view(name.back()) << " : ";
    printIntTypeRef(tag_width, Sign::kUnsigned);
  }

  if (!decl_only) {
    os << " {\n";
    for (const auto& term : type.terms()) {
      std::string term_name = term.name.str();
      os << term_name << " = " << term.tag << ",\n";
    }
    if (!has_value) {
      os << "\n};\n";
    } else {
      os << "} tag;\n";
      addVariantFunctions(type);
      os << "};\n";
    }
  } else {
    os << ";\n";
  }

  endNamespaceOf(name);
}

void SourceGenerator::addVariantBuilder(types::VariantType type, bool has_value,
                                        Width tag_width, bool is_external) {
  region_ = Region::kBuilders;

  auto name = type.cast<types::NominalType>().name();

  // Generate a BuildPJType specialization for this type.
  auto& os = stream();
  os << "namespace pj {\n";
  os << "namespace gen {\n";
  os << "template <>\n"
     << "struct BuildPJType<";
  printName(name);
  os << "> {\n";

  os << "static const auto* build(PJContext* ctx, const PJDomain* domain) {\n";

  os << "const PJTerm* terms[" << type.terms().size() << "];\n";
  size_t term_num = 0;
  for (const auto& term : type.terms()) {
    std::vector<llvm::StringRef> union_name{name.begin(), name.end()};
    auto term_handle = createTypeHandle({}, term.type);

    os << "terms[" << term_num++ << "] = PJCreateTerm(";
    os << "/*name=*/\"" << std::string_view(term.name) << "\"";
    os << ", /*type=*/" << term_handle;
    os << ", /*tag=*/" << term.tag << ");\n";
  }

  // Generate an array containing the name of the variant.
  auto name_array = buildStringArray(name);

  // Generate the final variant handle.
  std::string handle;
  std::string variant_type =
      type.isa<types::InlineVariantType>() ? "Inline" : "Outline";
  os << "const PJ" << variant_type << "VariantType* "
     << (handle = getUniqueName()) << " = PJCreate" << variant_type
     << "VariantType(ctx";
  os << ", /*name_size=*/" << name.size();
  os << ", /*name=*/" << name_array;
  os << ", /*type_domain=*/domain";
  os << ", /*num_terms=*/" << type.terms().size();
  os << ", /*terms=*/terms";
  os << ", /*default_term=*/" << type.default_term();
  if (domain_ == Domain::kHost) {
    assert(type.isa<types::InlineVariantType>());
    if (!has_value) {
      os << ", /*term_offset=*/-1, /*term_size=*/0";
    } else {
      os << ", /*term_offset=*/offsetof(";
      printName(name);
      os << ", value) << 3";
      os << ", /*term_size=*/sizeof(";
      printName(name);
      os << "::value) << 3";
    }
    if (!has_value) {
      os << ", /*tag_offset=*/0";
    } else {
      os << ", /*tag_offset=*/offsetof(";
      printName(name);
      os << ", tag) << 3";
    }
    os << ", /*tag_width=*/" << tag_width.bits();

    os << ", /*size=*/sizeof(";
    printName(name);
    os << ") << 3";

    os << ", /*alignment=*/alignof(";
    printName(name);
    os << ") << 3";
  } else {
    if (type.isa<types::InlineVariantType>()) {
      auto inline_var = type.cast<types::InlineVariantType>();
      os << ", /*term_offset=*/" << inline_var->term_offset.bits();
      os << ", /*term_size=*/" << inline_var->term_size.bits();
      os << ", /*tag_offset=*/" << inline_var->tag_offset.bits();
      os << ", /*tag_width=*/" << inline_var->tag_width.bits();
      os << ", /*size=*/" << inline_var->size.bits();
      os << ", /*alignment=*/" << inline_var->alignment.bits();
    } else {
      auto outline_var = type.cast<types::OutlineVariantType>();
      os << ", /*tag_width=*/" << outline_var->tag_width.bits();
      os << ", /*tag_alignment=*/" << outline_var->tag_alignment.bits();
      os << ", /*term_offset=*/" << outline_var->term_offset.bits();
      os << ", /*term_alignment=*/" << outline_var->term_alignment.bits();
    }
  }
  os << ");\n";

  os << "return " << handle << ";\n"
     << "}\n"
     << "};\n"
     << "}  // namespace gen\n"
     << "}  // namespace pj\n"
     << "\n";
}

void SourceGenerator::addVariant(types::VariantType type, bool is_external) {
  for (const auto& term : type.terms()) {
    addType(term.type);
  }

  const bool has_value = std::any_of(
      type.terms().begin(), type.terms().end(), [](const auto& term) {
        return !term.type.template isa<types::UnitType>();
      });

  pj::Width tag_width = compute_tag_width(type);

  if (domain_ == Domain::kHost && !is_external) {
    addVariantDef(type, has_value, tag_width, false);
  } else if (domain_ == Domain::kWire) {
    addVariantDef(type, has_value, tag_width, true);
  }

  addVariantBuilder(type, has_value, tag_width, is_external);
}

std::ostream& SourceGenerator::printDecoderSig(std::ostream& os,
                                               const std::string& name,
                                               const Portal::Decoder& decoder,
                                               bool is_cpp,
                                               bool is_declaration) {
  if (is_cpp) {
    os << "template <typename S";
    if (!is_declaration) {
      os << " = void";
    }
    os << ">";
  }
  std::string state = is_cpp ? "S" : "void";
  os << "BoundedBuffer " << name << "("
     << "const char* msg, " << getNameAsString(decoder.dst)
     << "* result, BoundedBuffer buffer,";
  if (is_cpp) {
    os << "std::array<::pj::DecodeHandler<" << getNameAsString(decoder.dst)
       << ", " << state << ">, " << decoder.handlers.size() << "> handlers, ";
  } else {
    os << "::pj::DecodeHandler<" << getNameAsString(decoder.dst) << ", "
       << state << "> handlers[], ";
  }
  os << state << "* state)";
  return os;
}

void SourceGenerator::addPortal(const SourceId& ns, const Portal& portal,
                                ParsedProtoFile::Protocol proto) {
  addProtocol(portal.proto, proto);

  // Define class in the header with the following interface.
  //
  // struct ClassName {
  //   Context* getContext();
  //
  //   // Returns the protocol in use (bound to the portal's context).
  //   Protocol getProtocol();
  //
  //   // Plans the protocol specified in the .pj file and JIT-compiles all
  //   // requested methods.
  //   ClassName();
  //
  //   // JIT-compiles all requested methods using the provided protocol.
  //   ClassName(const char* proto);
  //
  //   // Something like this depending on what was requested:
  //   size_t mySizeFn(const T* msg) {...}
  //   void myEncodeFn(const T* msg, char* buf) {...}
  //   template <typename S>
  //   BoundedBuffer myDecodeFn(
  //     const char* msg, T* result, BoundedBuffer buffer,
  //     ::pj::DecodeHandler<T, S> handlers[], S* state) {...}
  // };
  //
  // The class declaration goes into `defs_`, and the function bodies go into
  // `builders_`.
  defs_ << "#ifndef PROTOJIT_NO_INTERFACES\n";
  builders_ << "#ifndef PROTOJIT_NO_INTERFACES\n";

  region_ = Region::kDefs;
  beginNamespaceOf(ns);

  region_ = Region::kBuilders;
  beginNamespaceOf(ns);

  defs_ << "struct " << ns.back() << "{\n";

  auto generate_constructor_body = [&]() {
    assert(region_ == Region::kBuilders);

    std::stringstream adds;
    std::stringstream gets;

    for (auto& sizer : portal.sizers) {
      adds << "ctx_.addSizeFunction<" << getNameAsString(sizer.src) << ">(\""
           << sizer.name << "\", proto_, \"";
      sizer.src_path.print(adds);
      adds << "\", " << (sizer.round_up ? "true" : "false") << ");\n";

      gets << sizer.name << "_ = portal_.getSizeFunction<"
           << getNameAsString(sizer.src) << ">(\"" << sizer.name << "\");\n";
    }

    for (auto& encoder : portal.encoders) {
      adds << "ctx_.addEncodeFunction<" << getNameAsString(encoder.src)
           << ">(\"" << encoder.name << "\", proto_, \"";
      encoder.src_path.print(adds);
      adds << "\");\n";

      gets << encoder.name << "_ = portal_.getEncodeFunction<"
           << getNameAsString(encoder.src) << ">(\"" << encoder.name
           << "\");\n";
    }

    for (auto& decoder : portal.decoders) {
      // Generate in a new scope because we are declaring a variable.
      adds << "{\n";

      adds << "std::vector<std::string> handlers = {\n";
      for (auto& handler : decoder.handlers) {
        adds << "\"";
        handler.print(adds);
        adds << "\",";
      }
      adds << "};\n";

      adds << "ctx_.addDecodeFunction<" << getNameAsString(decoder.dst)
           << ">(\"" << decoder.name << "\", proto_, handlers);\n";

      adds << "}\n";  // End scope.

      gets << decoder.name << "_ = portal_.getDecodeFunction<"
           << getNameAsString(decoder.dst) << ">(\"" << decoder.name
           << "\");\n";
    }

    builders_ << adds.str() << "portal_ = ctx_.compile();\n" << gets.str();
  };

  // Generate no-argument constructor.
  defs_ << ns.back() << "();\n";

  region_ = Region::kBuilders;
  builders_ << ns.back() << "::" << ns.back() << "() : proto_(ctx_.plan<"
            << getNameAsString(proto.first.cast<types::NominalType>().name())
            << ">(\"" << proto.second << "\")) {\n";
  generate_constructor_body();
  builders_ << "}\n";

  // Generate constructor which takes an external protocol.
  defs_ << ns.back() << "(const char* proto);\n";

  region_ = Region::kBuilders;
  builders_ << ns.back() << "::" << ns.back()
            << "(const char* proto) : proto_(ctx_.decodeProto(proto)) {\n";
  generate_constructor_body();
  builders_ << "}\n";

  // Generate getProtocol() function.
  defs_ << "::pj::runtime::Protocol getProtocol();";
  builders_ << "::pj::runtime::Protocol " << ns.back() << "::getProtocol() {\n"
            << "return proto_;\n"
            << "}\n";

  // Generate getContext() function.
  defs_ << "::pj::runtime::Context* getContext();";
  builders_ << "::pj::runtime::Context* " << ns.back() << "::getContext() {\n"
            << "return &ctx_;\n"
            << "}\n";

  // Generate wrapper functions and member variables which they call into.
  std::stringstream fn_decls;
  std::stringstream vars;

  vars << "::pj::runtime::Context ctx_;\n"
       << "::pj::runtime::Protocol proto_;\n"
       << "::pj::runtime::Portal portal_;\n";

  for (auto& sizer : portal.sizers) {
    vars << "::pj::SizeFunction<" << getNameAsString(sizer.src) << ">"
         << sizer.name << "_;\n";

    fn_decls << "size_t " << sizer.name << "(const "
             << getNameAsString(sizer.src) << "* msg);\n";

    builders_ << "size_t " << ns.back() << "::" << sizer.name << "(const "
              << getNameAsString(sizer.src) << "* msg) {\n"
              << "return " << sizer.name << "_(msg);\n"
              << "}\n";
  }
  for (auto& encoder : portal.encoders) {
    vars << "::pj::EncodeFunction<" << getNameAsString(encoder.src) << ">"
         << encoder.name << "_;\n";

    fn_decls << "void " << encoder.name << "(const "
             << getNameAsString(encoder.src) << "* msg, char* buf);\n";

    builders_ << "void " << ns.back() << "::" << encoder.name << "(const "
              << getNameAsString(encoder.src) << "* msg, char* buf) {\n"
              << "return " << encoder.name << "_(msg, buf);\n"
              << "}\n";
  }
  for (auto& decoder : portal.decoders) {
    vars << "::pj::DecodeFunction<" << getNameAsString(decoder.dst) << ", void>"
         << decoder.name << "_;\n";

    printDecoderSig(fn_decls, decoder.name, decoder,
                    /*is_cpp=*/true, /*is_declaration=*/true)
        << ";\n";

    printDecoderSig(builders_, ns.back() + "::" + decoder.name, decoder,
                    /*is_cpp=*/true, /*is_declaration=*/false)
        << "{\n";
    builders_ << "return " << decoder.name
              << "_(msg, result, buffer, handlers.data(), state);\n"
              << "}\n";
  }

  defs_ << fn_decls.str() << "\n"
        << "private:\n"
        << vars.str();

  defs_ << "};\n";  // Terminate the struct.

  region_ = Region::kDefs;
  endNamespaceOf(ns);

  region_ = Region::kBuilders;
  endNamespaceOf(ns);

  defs_ << "#endif  // PROTOJIT_NO_INTERFACES\n";
  builders_ << "#endif  // PROTOJIT_NO_INTERFACES\n";
}

void SourceGenerator::addText(const SourceId& space, const std::string& text) {
  region_ = Region::kDefs;
  beginNamespaceOf(space, true);
  defs_ << text;
  endNamespaceOf(space, true);
}

void SourceGenerator::addPrecompilation(const SourceId& name,
                                        const Portal& portal,
                                        const SourceId& proto_name,
                                        ParsedProtoFile::Protocol proto) {
  addProtocol(proto_name, proto);

  // Define class with the following interface and generate cpp to precompile
  // methods for requested functions.
  //
  // struct ClassName {
  //   // Always present with this name for a precomp class.
  //   static std::string_view getSchema() {...}
  //
  //   // Always a zero-arg constructor.
  //   ClassName() {}
  //
  //   // Something like this depending on what was requested:
  //   size_t mySizeFn(const T* msg) {...}
  //   void myEncodeFn(const T* msg, char* buf) {...}
  //   template <typename S>
  //   BoundedBuffer myDecodeFn(
  //     const char* msg, T* result, BoundedBuffer buffer,
  //     ::pj::DecodeHandler<T, S> handlers[], S* state) {...}
  // }
  //
  // The class declaration goes into `defs_`, the function bodies go into
  // `builders_`, and the precompilation code goes into `cpp_`.
  defs_ << "#ifndef PROTOJIT_NO_INTERFACES\n";
  builders_ << "#ifndef PROTOJIT_NO_INTERFACES\n";

  region_ = Region::kDefs;
  beginNamespaceOf(name);

  region_ = Region::kBuilders;
  beginNamespaceOf(name);

  region_ = Region::kCpp;

  defs_ << "struct " << name.back() << "{\n" << name.back() << "() {}\n";

  {
    auto schema_ptr_name = "protocol_ptr_" + getNameAsString(name, "_");
    auto schema_size_name = "protocol_size_" + getNameAsString(name, "_");

    defs_ << "static std::string_view getSchema();\n";

    builders_ << "extern \"C\" const char " << schema_ptr_name << ";\n"
              << "extern \"C\" size_t " << schema_size_name << ";\n"
              << "std::string_view  " << name.back() << "::getSchema() {\n"
              << "return {&" << schema_ptr_name << ", " << schema_size_name
              << "};\n"
              << "}\n";

    cpp_ << "ctx_.addProtocolDefinition(\"" << schema_ptr_name << "\", \""
         << schema_size_name << "\", " << getNameAsString(proto_name, "_")
         << ");\n";
  }

  for (auto& sizer : portal.sizers) {
    defs_ << "size_t " << sizer.name << "(const " << getNameAsString(sizer.src)
          << "* msg);\n";

    auto sizer_name = getNameAsString(name, "_") + "_" + sizer.name;

    builders_ << "extern \"C\" size_t " << sizer_name << "(const void*);\n";
    builders_ << "size_t " << name.back() << "::" << sizer.name << "(const "
              << getNameAsString(sizer.src) << "* msg) {\n"
              << "return " << sizer_name << "(msg);\n"
              << "}\n";

    cpp_ << "ctx_.addSizeFunction<" << getNameAsString(sizer.src) << ">(\""
         << sizer_name << "\", " << getNameAsString(proto_name, "_") << ", \"";
    sizer.src_path.print(cpp_);
    cpp_ << "\", " << (sizer.round_up ? "true" : "false") << ");\n";
  }

  for (auto& encoder : portal.encoders) {
    defs_ << "void " << encoder.name << "(const "
          << getNameAsString(encoder.src) << "*, char* buf);\n";

    auto encoder_name = getNameAsString(name, "_") + "_" + encoder.name;

    builders_ << "extern \"C\" void " << encoder_name << "(const "
              << getNameAsString(encoder.src) << "*, char* buf);\n";
    builders_ << "void " << name.back() << "::" << encoder.name << "(const "
              << getNameAsString(encoder.src) << "* msg, char* buf) {\n"
              << "" << encoder_name << "(msg, buf);\n"
              << "}\n";

    cpp_ << "ctx_.addEncodeFunction<" << getNameAsString(encoder.src) << ">(\""
         << encoder_name << "\", " << getNameAsString(proto_name, "_")
         << ", \"";
    encoder.src_path.print(cpp_);
    cpp_ << "\");\n";
  }

  for (auto& decoder : portal.decoders) {
    printDecoderSig(defs_, decoder.name, decoder, /*is_cpp=*/true,
                    /*is_declaration=*/true)
        << ";";

    auto decoder_name = getNameAsString(name, "_") + "_" + decoder.name;

    builders_ << "extern \"C\" ";
    printDecoderSig(builders_, decoder_name, decoder,
                    /*is_cpp=*/false, /*is_declaration=*/false)
        << ";\n";

    printDecoderSig(builders_, name.back() + "::" + decoder.name, decoder,
                    /*is_cpp=*/true, /*is_declaration=*/false)
        << "{\n";
    builders_ << "return " << decoder_name
              << "(msg, result, buffer, handlers.data(), state);\n"
              << "}\n";

    // Generate cpp in a new scope because we are declaring a variable.
    cpp_ << "{\n";

    cpp_ << "std::vector<std::string> handlers = {\n";
    for (auto& handler : decoder.handlers) {
      cpp_ << "\"";
      handler.print(cpp_);
      cpp_ << "\",";
    }
    cpp_ << "};\n";

    cpp_ << "ctx_.addDecodeFunction<" << getNameAsString(decoder.dst) << ">(\""
         << decoder_name << "\", " << getNameAsString(proto_name, "_")
         << ", handlers);\n";

    cpp_ << "}\n";  // End scope.
  }

  defs_ << "};\n";  // Terminate the struct.

  region_ = Region::kDefs;
  endNamespaceOf(name);

  region_ = Region::kBuilders;
  endNamespaceOf(name);

  defs_ << "#endif  // PROTOJIT_NO_INTERFACES\n";
  builders_ << "#endif  // PROTOJIT_NO_INTERFACES\n";
}

void SourceGenerator::generate(
    const std::filesystem::path& path, std::ostream& output, std::ostream* cpp,
    const std::vector<std::filesystem::path>& imports) {
  output << "#pragma once\n"
         << "#include <cstddef>\n"
         << "#include <string_view>\n"
         << "#include \"pj/runtime.h\"\n"
         << "#include \"pj/traits.hpp\"\n"
         << "\n";

  for (auto& import : imports) {
    output << "#include \"" << import.c_str() << ".hpp\"\n";
  }

  for (auto& header : base_headers_) {
    output << "#include \"pj/" << header << ".hpp\"\n";
  }

  output << defs_.str();
  output << builders_.str();

  if (cpp) {
    *cpp << "#include <cstdio>\n"
         << "#include <cstring>\n"
         << "#define PROTOJIT_NO_INTERFACES\n"
         << "#include \"pj/runtime.hpp\"\n"
         << "#include " << path.filename() << "\n"
         << "int main(int argc, char** argv) {\n"
         << "  if (argc != 3) {\n"
         << "    fprintf(stderr, \"Usage: <precompiler> o/pic "
            "<filename>\\n\");\n"
         << "    return 1;\n"
         << "  }\n"
         << "  pj::runtime::Context ctx_;\n"
         << cpp_.str() << "\n"
         << "  ctx_.precompile(argv[2], !strcmp(argv[1], \"pic\"));\n"
         << "}\n";
  }
};

}  // namespace pj
