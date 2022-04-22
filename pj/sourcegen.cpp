#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "protogen.hpp"
#include "types.hpp"

namespace pj {

std::string getUniqueID() {
  static size_t counter = 0;
  return std::to_string(counter++);
}

void generateNameRef(types::Name name, std::ostream& output) {
  for (uintptr_t i = 0; i < name.size(); ++i) {
    output << std::string_view(name[i]);
    if (i < name.size() - 1) {
      output << "::";
    }
  }
}

template <typename T>
void generateNamespaceBegin(const T& name, std::ostream& output) {
  for (uintptr_t i = 0; i < name.size() - 1; ++i) {
    output << "namespace " << std::string_view(name[i]) << "{";
  }
}

template <typename T>
void generateNamespaceEnd(const T& name, std::ostream& output) {
  for (uintptr_t i = 0; i < name.size() - 1; ++i) {
    output << "}\n";
  }
  output << '\n';
}

// TODO: platform-specific
std::array<std::string_view, 4> kIntTypes{"char", "short", "int", "long"};

pj::Width generateIntTypeRef(pj::Width width, types::Int::Sign sign,
                             std::ostream& output) {
  assert(width.bytes() > 0);
  if (sign == types::Int::kSignless) {
    // TODO: maybe use wchar_t?
  }
  if (sign == types::Int::kUnsigned) {
    output << "unsigned ";
  }
  auto log = static_cast<uintptr_t>(std::ceil(std::log2(width.bytes())));
  assert(log <= kIntTypes.size());
  output << kIntTypes[log];

  return pj::Bytes(1 << log);
}

void generateTypeRef(mlir::Type type, std::ostream& output) {
  if (auto named = type.dyn_cast<types::NominalType>()) {
    generateNameRef(named.name(), output);
  } else if (auto I = type.dyn_cast<types::IntType>()) {
    generateIntTypeRef(I->width, I->sign, output);
  } else if (auto A = type.dyn_cast<types::ArrayType>()) {
    output << "std::array<";
    generateTypeRef(A->elem, output);
    output << ", " << A->length << ">";
  } else if (auto V = type.dyn_cast<types::VectorType>()) {
    output << "pj::ArrayView<";
    generateTypeRef(V->elem, output);
    output << ", " << V->min_length << ", " << V->max_length << ">";
  } else {
    UNREACHABLE();
  }
}

std::string generateTypeRef(mlir::Type type) {
  std::ostringstream o;
  generateTypeRef(type, o);
  return o.str();
}

template <typename T>
void printNamespacedName(const T& name, std::ostream& output) {
  for (auto& p : name) output << "::" << std::string_view(p);
}

std::pair<std::string, std::string> buildPJVariableDecl(mlir::Type type,
                                                        std::ostream& output) {
  std::string rt_type;
  std::string var;
  if (!type) {
    output << (rt_type = "const PJUnitType*") << " "
           << (var = "unit" + getUniqueID());
  } else if (type.isa<types::StructType>()) {
    output << (rt_type = "const PJStructType*") << " "
           << (var = "struct" + getUniqueID());
  } else if (type.isa<types::InlineVariantType>()) {
    output << (rt_type = "const PJInlineVariantType*") << " "
           << (var = "inline_variant" + getUniqueID());
  } else if (type.isa<types::OutlineVariantType>()) {
    // OutlineVariants only exist on the wire
    UNREACHABLE();
  } else if (type.isa<types::IntType>()) {
    output << (rt_type = "const PJIntType*") << " "
           << (var = "int" + getUniqueID());
  } else if (type.isa<types::ArrayType>()) {
    output << (rt_type = "const PJArrayType*") << " "
           << (var = "arr" + getUniqueID());
  } else if (type.isa<types::VectorType>()) {
    output << (rt_type = "const PJVectorType*") << " "
           << (var = "vector" + getUniqueID());
  } else {
    UNREACHABLE();
  }
  return std::make_pair(rt_type, var);
}

std::string buildTypeGeneratorStmt(mlir::Type type, std::ostream& output) {
  if (!type) {
    // Normalize null types to the unit type for the rest of ProtoJIT.
    // These are represented differently from an empty struct in the
    // source generator because in C++ an empty struct has size 8 bits,
    // not 0 bits.
    auto [_, var] = buildPJVariableDecl(type, output);
    output << " = PJCreateUnitType(ctx);";
    return var;
  }

  std::string type_ref = generateTypeRef(type);
  if (auto named = type.dyn_cast<types::NominalType>()) {
    auto [rt_type, var] = buildPJVariableDecl(type, output);
    output << " = static_cast<" << rt_type << ">(BuildPJType<";
    printNamespacedName(named.name(), output);
    output << ">::build(ctx));\n";
    return var;
  }

  if (auto I = type.dyn_cast<types::IntType>()) {
    auto [_, var] = buildPJVariableDecl(type, output);
    output << " = PJCreateIntType(ctx";
    output << ", /*width=*/" << I->width.bits();
    {
      output << ", /*alignment=*/alignof(";
      pj::Width actual_width = generateIntTypeRef(I->width, I->sign, output);
      output << ") << 3";
      // TODO: come back to this when implementing bitfields
      ASSERT(I->width == actual_width);
    }
    output << ", /*sign=*/";
    switch (I->sign) {
      case types::Int::kSigned:
        output << "PJ_SIGN_SIGNED);\n";
        break;
      case types::Int::kUnsigned:
        output << "PJ_SIGN_UNSIGNED);\n";
        break;
      case types::Int::kSignless:
        output << "PJ_SIGN_SIGNLESS);\n";
        break;
      default:
        UNREACHABLE();
    }
    return var;
  }

  if (auto A = type.dyn_cast<types::ArrayType>()) {
    std::string elem_type_ref = generateTypeRef(A->elem);
    std::string elem_var = buildTypeGeneratorStmt(A->elem, output);
    auto [_, var] = buildPJVariableDecl(type, output);
    output << " = PJCreateArrayType(ctx";
    output << ", /*elem=*/" << elem_var;
    output << ", /*length=*/" << A->length;
    output << ", /*elem_size=*/sizeof(" << elem_type_ref << ") << 3";
    output << ", /*alignment=*/alignof(" << type_ref << ") << 3);\n";
    return var;
  }

  if (auto V = type.dyn_cast<types::VectorType>()) {
    std::string using_decl = "Vec" + getUniqueID();
    output << "using " << using_decl << " = " << type_ref << ";\n";

    std::string elem_var = buildTypeGeneratorStmt(V->elem, output);
    auto [_, var] = buildPJVariableDecl(type, output);
    output << " = PJCreateVectorType(ctx";
    output << ", /*elem=*/" << elem_var;
    output << ", /*min_length=*/" << V->min_length;
    output << ", /*max_length=*/" << V->max_length;
    output << ", /*wire_min_length=*/" << V->min_length;
    output << ", /*ppl_count=*/0";
    output << ", /*length_offset=*/offsetof(" << using_decl << ", length) << 3";
    output << ", /*length_size=*/sizeof(" << using_decl << "::length) << 3";
    output << ", /*ref_offset=*/offsetof(" << using_decl << ", outline) << 3";
    output << ", /*ref_size=*/sizeof(" << using_decl << "::outline) << 3";
    output << ", /*reference_mode=*/PJ_REFERENCE_MODE_POINTER";
    if (V->min_length > 0) {
      output << ", /*inline_payload_offset=*/offsetof(" << using_decl
             << ", storage) << 3";
      output << ", /*inline_payload_size=*/sizeof(" << using_decl
             << "::storage) << 3";
    } else {
      output << ", /*inline_payload_offset=*/-1";
      output << ", /*inline_payload_size=*/0";
    }
    output << ", /*partial_payload_offset=*/-1";
    output << ", /*partial_payload_size=*/0";
    output << ", /*size=*/sizeof(" << using_decl << ") << 3";
    output << ", /*alignment=*/alignof(" << using_decl << ") << 3";
    output << ", /*outlined_payload_alignment=*/64);\n";
    return var;
  }

  UNREACHABLE();
}

void buildCStringArray(Span<llvm::StringRef> arr, std::string_view var,
                       std::ostream& output) {
  output << "const char* " << var << "[" << arr.size() << "] = {";
  for (uintptr_t i = 0; i < arr.size(); ++i) {
    output << "\"" << std::string_view(arr[i]) << "\"";
    if (i < arr.size() - 1) {
      output << ", ";
    }
  }
  output << "};\n";
}

void generateVariant(const ParsedProtoFile::Decl& decl, std::ostream& output,
                     std::ostream& back) {
  auto type = decl.type.cast<types::InlineVariantType>();

  auto nominal = type.template cast<types::NominalType>();
  assert(nominal);

  auto name = nominal.name();

  const bool has_value =
      std::any_of(type->terms.begin(), type->terms.end(),
                  [](auto& term) { return bool(term.type); });

  generateNamespaceBegin(name, output);

  if (!decl.is_enum) {
    output << "struct " << std::string_view(name.back()) << " {\n";
    if (has_value) {
      output << "union {\n";
      for (auto& term : type->terms) {
        if (term.type) {
          generateTypeRef(term.type, output);
          output << " " << std::string_view(term.name) << ";\n";
        }
      }
      output << "} value;\n";
    }
  }

  // Define an enum class with all options.
  if (decl.is_enum) {
    output << "enum class " << std::string_view(name.back()) << " : ";
  } else {
    output << "enum class Kind : ";
  }
  pj::Width tag_width =
      generateIntTypeRef(compute_tag_width(types::InlineVariant(type)),
                         types::Int::Sign::kUnsigned, output);
  output << " {\n";
  output << "undef = " << 0 << ",\n";
  for (auto& term : type->terms) {
    std::string term_name = term.name.str();
    output << term_name << " = " << term.tag << ",\n";
  }

  if (decl.is_enum) {
    output << "\n};\n";
  } else {
    output << "} tag;\n};\n";
  }
  generateNamespaceEnd(name, output);

  // Generate a BuildPJType specialization for this type.
  back << "namespace pj {\n";
  back << "namespace gen {\n";
  back << "template <>\n"
       << "struct BuildPJType<";
  printNamespacedName(name, back);
  back << "> {\n";

  back << "static const void* build(PJContext* ctx) {\n";

  back << "const PJTerm* terms[" << type->terms.size() << "];\n";
  uintptr_t term_num = 0;
  for (auto& term : type->terms) {
    std::string var = buildTypeGeneratorStmt(term.type, back);

    back << "terms[" << term_num++ << "] = PJCreateTerm(";
    back << "/*name=*/\"" << std::string_view(term.name) << "\"";
    back << ", /*type=*/" << var;
    back << ", /*tag=*/" << term.tag << ");\n";
  }

  buildCStringArray(name, "name", back);

  auto [_, variant_var] = buildPJVariableDecl(type, back);
  back << " = PJCreateInlineVariantType(ctx";
  back << ", /*name_size=*/" << name.size();
  back << ", /*name=*/name";
  back << ", /*type_domain=*/"
       << (nominal.type_domain() == types::TypeDomain::kHost
               ? "PJ_TYPE_DOMAIN_HOST"
               : "PJ_TYPE_DOMAIN_WIRE");
  back << ", /*num_terms=*/" << type->terms.size();
  back << ", /*terms=*/terms";
  if (decl.is_enum || !has_value) {
    back << ", /*term_offset=*/-1, /*term_size=*/0";
  } else {
    back << ", /*term_offset=*/offsetof(";
    printNamespacedName(name, back);
    back << ", value) << 3";
    back << ", /*term_size=*/sizeof(";
    printNamespacedName(name, back);
    back << "::value) << 3";
  }
  if (decl.is_enum) {
    back << ", /*tag_offset=*/0";
  } else {
    back << ", /*tag_offset=*/offsetof(";
    printNamespacedName(name, back);
    back << ", tag) << 3";
  }
  back << ", /*tag_width=*/" << tag_width.bits();
  {
    back << ", /*size=*/sizeof(";
    printNamespacedName(name, back);
    back << ") << 3";
  }
  {
    back << ", /*alignment=*/alignof(";
    printNamespacedName(name, back);
    back << ") << 3";
  }
  back << ");\n";

  back << "return " << variant_var << ";\n"
       << "}\n"
       << "};\n"
       << "}  // namespace gen\n"
       << "}  // namespace pj\n"
       << "\n";
}

void generateStruct(const ParsedProtoFile::Decl& decl, std::ostream& output,
                    std::ostream& back) {
  auto type = decl.type.cast<types::StructType>();
  assert(type);

  auto nominal = type.cast<types::NominalType>();
  assert(nominal);

  auto name = nominal.name();

  generateNamespaceBegin(name, output);

  output << "struct " << std::string_view(name.back()) << " {\n";

  for (auto& field : type->fields) {
    generateTypeRef(field.type, output);
    output << " " << std::string_view(field.name) << ";\n";
  }
  output << "};\n\n";

  generateNamespaceEnd(name, output);

  // Generate a BuildPJType specialization for this type.
  back << "namespace pj {\n";
  back << "namespace gen {\n";
  back << "template <>\n"
       << "struct BuildPJType<";
  printNamespacedName(name, back);
  back << "> {\n";

  // For each field:
  //   - Build an MLIR type for this field:
  //     - either inline for primitive types
  //     - or a call to a specialization of BuildPJType for a composite type

  back << "static const void* build(PJContext* ctx) {\n";

  back << "const PJStructField* fields[" << type->fields.size() << "];\n";
  uintptr_t field_num = 0;
  for (auto& field : type->fields) {
    std::string var = buildTypeGeneratorStmt(field.type, back);

    back << "fields[" << field_num++ << "] = PJCreateStructField(";
    back << "/*name=*/\"" << std::string_view(field.name) << "\"";
    back << ", /*type=*/" << var;
    {
      back << ", /*offset=*/offsetof(";
      printNamespacedName(name, back);
      back << ", " << std::string_view(field.name) << ") << 3";
    }
    back << ");\n";
  }

  buildCStringArray(name, "name", back);

  auto [_, struct_var] = buildPJVariableDecl(type, back);
  back << " = PJCreateStructType(ctx";
  back << ", /*name_size=*/" << name.size();
  back << ", /*name=*/name";
  back << ", /*type_domain=*/"
       << (nominal.type_domain() == types::TypeDomain::kHost
               ? "PJ_TYPE_DOMAIN_HOST"
               : "PJ_TYPE_DOMAIN_WIRE");
  back << ", /*num_fields=*/" << type->fields.size();
  back << ", /*fields=*/fields";
  {
    back << ", /*size=*/sizeof(";
    printNamespacedName(name, back);
    back << ") << 3";
  }
  {
    back << ", /*alignment=*/alignof(";
    printNamespacedName(name, back);
    back << ") << 3";
  }
  back << ");\n";

  back << "return " << struct_var << ";\n"
       << "}\n"
       << "};\n"
       << "}  // namespace gen\n"
       << "}  // namespace pj\n"
       << "\n";
}

void generateComposite(ParsedProtoFile::Decl decl, std::ostream& output,
                       std::ostream& back) {
  auto type = decl.type;

  if (type.isa<types::StructType>()) {
    generateStruct(decl, output, back);
  } else if (type.isa<types::InlineVariantType>()) {
    generateVariant(decl, output, back);
  } else {
    UNREACHABLE();
  }
};

void generateTypedef(const ParsedProtoFile::Decl& decl, std::ostream& output) {
  generateNamespaceBegin(decl.name, output);
  output << "using " << std::string_view(decl.name.back());
  output << " = ";
  generateTypeRef(decl.type, output);
  output << ";\n";
  generateNamespaceEnd(decl.name, output);
}

void generateProtocol(const SourceId& name, mlir::Type head,
                      const std::optional<types::PathAttr>& tag_path,
                      std::ostream& output, std::ostream& back) {
  generateNamespaceBegin(name, output);
  output << "struct " << std::string_view(name.back()) << ";\n";
  generateNamespaceEnd(name, output);

  back << "namespace pj {\n";
  back << "namespace gen {\n";
  back << "template <>\n"
       << "struct ProtocolHead<";
  printNamespacedName(name, back);
  back << "> {\n"
       << "using Head = ";
  generateTypeRef(head, back);
  back << ";\n";
  back << "static std::string tag() { return \"";
  if (tag_path.has_value()) {
    back << tag_path->toString() << "\"; }\n";
  } else {
    back << "\"; }\n";
  }
  back << "};\n}  // namespace gen\n\n";
  back << "\n}  // namespace pj\n\n";
}

void generateHeader(const ArchDetails& arch, const ParsedProtoFile& file,
                    std::ostream& output) {
  output << "#pragma once\n"
         << "#include <cstddef>\n "
         << "#include \"pj/protojit.hpp\"\n"
         << "#include \"pj/runtime.h\"\n"
         << "\n";

  for (auto& import : file.imports) {
    output << "#include \"" << import.c_str() << ".hpp\"\n";
  }

  std::ostringstream back;
  // TODO: setup namespaces outside
  for (auto& decl : file.decls) {
    switch (decl.kind) {
      case ParsedProtoFile::DeclKind::kType:
        generateTypedef(decl, output);
        break;
      case ParsedProtoFile::DeclKind::kComposite:
        generateComposite(decl, output, back);
        break;
      case ParsedProtoFile::DeclKind::kProtocol:
        generateProtocol(decl.name, decl.type, decl.tag_path, output, back);
        break;
    }
  }
  output << back.str();
};

}  // namespace pj
