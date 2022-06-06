#include <algorithm>
#include <charconv>
#include <cmath>
#include <unordered_map>

#include "defer.hpp"
#include "protogen.hpp"
#include "sourcegen.hpp"
#include "types.hpp"

namespace pj {

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

void SourceGenerator::printIntTypeRef(Width width, Sign sign) {
  assert(width.bytes() > 0);

  auto& os = stream();
  if (region_ == Region::kBuilders) os << "Integer<";

  if (sign == Sign::kSignless) {
    // TODO: maybe use wchar_t?
  }
  if (sign == Sign::kUnsigned) os << "unsigned ";

  auto log = static_cast<uintptr_t>(std::ceil(std::log2(width.bytes())));

  // TODO: platform-specific
  std::array<std::string_view, 4> kIntTypes{"char", "short", "int", "long"};
  assert(log <= kIntTypes.size());
  os << kIntTypes[log];

  if (region_ == Region::kBuilders) os << ", " << convertSign(sign) << ">";
}

void SourceGenerator::printTypeRef(types::ValueType type) {
  auto& os = stream();

  if (auto named = type.dyn_cast<types::NominalType>()) {
    printName(named.name());
    return;
  }

  if (auto I = type.dyn_cast<types::IntType>()) {
    printIntTypeRef(I->width, I->sign);
    return;
  }

  if (auto U = type.dyn_cast<types::UnitType>()) {
    os << "pj::Unit";
    return;
  }

  if (auto A = type.dyn_cast<types::ArrayType>()) {
    os << "std::array<";
    printTypeRef(A->elem);
    os << ", " << A->length << ">";
    return;
  }

  if (auto V = type.dyn_cast<types::VectorType>()) {
    os << "pj::ArrayView<";
    printTypeRef(V->elem);
    os << ", " << V->min_length << ", " << V->max_length << ">";
    return;
  }

  UNREACHABLE();
}

std::string SourceGenerator::createTypeHandle(types::ValueType type) {
  assert(region_ == Region::kBuilders);
  auto& os = stream();

  std::string handle;

  // Allow protojit.hpp to generate the layout based on the host compiler's
  // layout decisions if it's for a host type. BuildPJType is always used for
  // NominalTypes under the assumption that sourcegen is not used for host and
  // wire types at once.
  if (domain_ == Domain::kHost || type.isa<types::NominalType>()) {
    std::string handle_type = [&]() {
      if (type.isa<types::StructType>()) return "const PJStructType*";
      if (type.isa<types::InlineVariantType>())
        return "const PJInlineVariantType*";
      if (type.isa<types::OutlineVariantType>())
        return "const PJOutlineVariantType*";
      if (type.isa<types::IntType>()) return "const PJIntType*";
      if (type.isa<types::UnitType>()) return "const PJUnitType*";
      if (type.isa<types::ArrayType>()) return "const PJArrayType*";
      if (type.isa<types::VectorType>()) return "const PJVectorType*";
      UNREACHABLE();
    }();

    os << "const auto* " << (handle = getUniqueName()) << " = static_cast<"
       << handle_type << ">(BuildPJType<";
    printTypeRef(type);
    os << ">::build(ctx));\n";
    return handle;
  }

  // Copy the type information exactly as-is.
  if (auto I = type.dyn_cast<types::IntType>()) {
    os << "const auto* " << (handle = getUniqueName())
       << " = PJCreateIntType(ctx";
    os << ", /*width=*/" << I->width.bits();
    os << ", /*alignment=*/" << I->alignment.bits();
    os << ", /*sign=*/" << convertSign(I->sign) << ");\n";
    return handle;
  }

  if (auto U = type.dyn_cast<types::UnitType>()) {
    os << "const PJUnitType* " << (handle = getUniqueName())
       << " = PJCreateUnitType(ctx);\n";
    return handle;
  }

  if (auto A = type.dyn_cast<types::ArrayType>()) {
    std::string elem_handle = createTypeHandle(A->elem);
    os << "const auto* " << (handle = getUniqueName())
       << " = PJCreateArrayType(ctx";
    os << ", /*elem=*/" << elem_handle;
    os << ", /*length=*/" << A->length;
    os << ", /*elem_size=*/" << A->elem_size.bits();
    os << ", /*alignment=*/" << A->alignment.bits();
    return elem_handle;
  }

  if (auto V = type.dyn_cast<types::VectorType>()) {
    std::string elem_handle = createTypeHandle(V->elem);
    os << "const auto* " << (handle = getUniqueName())
       << " = PJCreateVectorType(ctx";
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
        os << "PJ_REFERENCE_MODE_OFFSET);\n";
        break;
      case ReferenceMode::kPointer:
        os << "PJ_REFERENCE_MODE_POINTER);\n";
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
       << V->outlined_payload_alignment.bits();
    return handle;
  }

  UNREACHABLE();
}

std::string SourceGenerator::buildStringArray(Span<llvm::StringRef> arr) {
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

void SourceGenerator::addProtocolHead(const SourceId& name,
                                      types::ValueType type,
                                      types::PathAttr tag_path) {
  if (shouldAdd(type)) {
    addComposite(type);
  }

  region_ = Region::kDefs;
  beginNamespaceOf(name);
  stream() << "struct " << std::string_view(name.back()) << ";\n";
  endNamespaceOf(name);

  region_ = Region::kBuilders;
  auto& os = stream();
  os << "namespace pj {\n";
  os << "namespace gen {\n";
  os << "template <>\n"
     << "struct ProtocolHead<";
  printName(name);
  os << "> {\n"
     << "using Head = ";
  printTypeRef(type);
  os << ";\n";
  os << "static std::string tag() { return \"" << tag_path.toString()
     << "\"; }\n";
  os << "};\n}  // namespace gen\n\n";
  os << "\n}  // namespace pj\n\n";
}

void SourceGenerator::addProtocol(const SourceId& name,
                                  types::ProtocolType proto) {
  pushDomain(Domain::kWire);

  if (shouldAdd(proto->head)) {
    addComposite(proto->head);
  }

  region_ = Region::kDefs;
  beginNamespaceOf(name);
  stream() << "struct " << std::string_view(name.back()) << ";\n";
  endNamespaceOf(name);

  region_ = Region::kBuilders;
  auto& os = stream();
  os << "namespace pj {\n";
  os << "namespace gen {\n";
  os << "template <>\n"
     << "struct BuildPJType<";
  printName(name);
  os << "> {\n"
     << "using Head = ";
  printTypeRef(proto->head);
  os << ";\n";

  os << "static const void* build(PJContext* ctx) {\n";
  std::string head_handle = createTypeHandle(proto->head);
  os << "return PJCreateProtocolType(ctx, " << head_handle << ", "
     << proto->buffer_offset.bits() << ");\n";
  os << "}\n";

  os << "};\n}  // namespace gen\n\n";
  os << "\n}  // namespace pj\n\n";

  popDomain();
}

void SourceGenerator::addComposite(types::ValueType type, bool is_external) {
  pushDomain(type.cast<types::NominalType>().type_domain());

  if (type.isa<types::StructType>()) {
    addStruct(type.cast<types::StructType>(), is_external);
  } else {
    addVariant(type.cast<types::VariantType>(), is_external);
  }

  generated_.insert(type.getAsOpaquePointer());
  popDomain();
}

void SourceGenerator::addStructDef(types::StructType type) {
  region_ = Region::kDefs;

  auto name = type.cast<types::NominalType>().name();
  beginNamespaceOf(name);

  stream() << "struct " << std::string_view(name.back()) << " {\n";
  for (auto& field : type->fields) {
    printTypeRef(field.type);
    stream() << " " << std::string_view(field.name) << ";\n";
  }
  stream() << "};\n\n";

  endNamespaceOf(name);
}

void SourceGenerator::addStructBuilder(types::StructType type) {
  region_ = Region::kBuilders;

  auto& os = stream();
  auto name = type.cast<types::NominalType>().name();

  os << "namespace pj {\n";
  os << "namespace gen {\n";
  os << "template <>\n"
     << "struct BuildPJType<";
  printName(name);
  os << "> {\n";

  os << "static const void* build(PJContext* ctx) {\n";

  // Generate an array of handles for each of the fields.
  os << "const PJStructField* fields[" << type->fields.size() << "];\n";
  uintptr_t field_num = 0;
  for (auto& field : type->fields) {
    std::string field_handle = createTypeHandle(field.type);

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
  os << ", /*type_domain=*/"
     << (domain_ == Domain::kHost ? "PJ_TYPE_DOMAIN_HOST"
                                  : "PJ_TYPE_DOMAIN_WIRE");
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
    if (shouldAdd(field.type)) {
      addComposite(field.type);
    }
  }

  if (domain_ == Domain::kHost && !is_external) {
    addStructDef(type);
  }

  addStructBuilder(type);
}

void SourceGenerator::addVariantDef(types::VariantType type, bool has_value,
                                    Width tag_width) {
  assert(type.isa<types::InlineVariantType>());

  region_ = Region::kDefs;

  auto& os = stream();
  auto name = type.cast<types::NominalType>().name();

  beginNamespaceOf(name);

  if (has_value) {
    os << "struct " << std::string_view(name.back()) << " {\n";
    if (has_value) {
      os << "union {\n";
      for (const auto& term : type.terms()) {
        if (term.type) {
          printTypeRef(term.type);
          os << " " << std::string_view(term.name) << ";\n";
        }
      }
      os << "} value;\n";
    }
  }

  // Define an enum class with all options.
  if (!has_value) {
    os << "enum class " << std::string_view(name.back()) << " : ";
  } else {
    os << "enum class Kind : ";
  }
  printIntTypeRef(tag_width, Sign::kUnsigned);

  os << " {\n";
  os << "undef = " << 0 << ",\n";
  for (const auto& term : type.terms()) {
    std::string term_name = term.name.str();
    os << term_name << " = " << term.tag << ",\n";
  }
  if (!has_value) {
    os << "\n};\n";
  } else {
    os << "} tag;\n};\n";
  }

  endNamespaceOf(name);
}

void SourceGenerator::addVariantBuilder(types::VariantType type, bool has_value,
                                        Width tag_width) {
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

  os << "static const void* build(PJContext* ctx) {\n";

  os << "const PJTerm* terms[" << type.terms().size() << "];\n";
  size_t term_num = 0;
  for (const auto& term : type.terms()) {
    std::string term_handle = createTypeHandle(term.type);

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
  os << ", /*type_domain=*/"
     << (domain_ == Domain::kHost ? "PJ_TYPE_DOMAIN_HOST"
                                  : "PJ_TYPE_DOMAIN_WIRE");
  os << ", /*num_terms=*/" << type.terms().size();
  os << ", /*terms=*/terms";
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
    if (shouldAdd(term.type)) {
      addComposite(term.type);
    }
  }

  const bool has_value = std::any_of(
      type.terms().begin(), type.terms().end(), [](const auto& term) {
        return !term.type.template isa<types::UnitType>();
      });

  pj::Width tag_width = compute_tag_width(type);

  if (domain_ == Domain::kHost && !is_external) {
    addVariantDef(type, has_value, tag_width);
  }

  addVariantBuilder(type, has_value, tag_width);
}

void SourceGenerator::generateHeader(
    std::ostream& output, const std::vector<std::filesystem::path>& imports) {
  output << "#pragma once\n"
         << "#include <cstddef>\n"
         << "#include \"pj/protojit.hpp\"\n"
         << "#include \"pj/runtime.h\"\n"
         << "\n";

  for (auto& import : imports) {
    output << "#include \"" << import.c_str() << ".hpp\"\n";
  }

  output << defs_.str();
  output << builders_.str();
};

}  // namespace pj
