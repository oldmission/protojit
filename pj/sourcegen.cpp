#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "concrete_types.hpp"
#include "protogen.hpp"

namespace pj {

void PlanMemory(Scope* scope, ParsedProtoFile& file) {
  std::unordered_map<const AType*, const CType*> memo;

  for (auto& decl : file.decls) {
#if 0
    if (decl.kind == ParsedProtoFile::DeclKind::kProtocol) {
      continue;
    }
#endif

    // We can't call PlanMemo directly here, because the type inside
    // may itself be a named type, in the event of a typedef.
    auto inner = decl.atype->AsNamed()->named;
    const CType* type = nullptr;
    if (inner->IsVariant() && !decl.explicit_tags.empty()) {
      type = inner->AsVariant()->PlanWithTags(scope, memo, decl.explicit_tags);
    } else if (inner->IsStruct()) {
      type =
          inner->AsStruct()->PlanWithFieldOrder(scope, memo, decl.field_order);
    } else {
      type = decl.atype->AsNamed()->named->Plan(scope, memo);
    }
    type = new (scope) CNamedType(decl.atype->AsNamed(), type);
    memo[decl.atype] = type;
    decl.ctype = type;
  }
}

void GenerateSourceIdRef(const SourceId& source, std::ostream& output) {
  for (intptr_t i = 0; i < source.size(); ++i) {
    output << source[i];
    if (i < source.size() - 1) {
      output << "::";
    }
  }
}

void GenerateNamespaceBegin(const SourceId& source, std::ostream& output) {
  for (intptr_t i = 0; i < source.size() - 1; ++i) {
    output << "namespace " << source[i] << "{";
  }
}

void GenerateNamespaceEnd(const SourceId& source, std::ostream& output) {
  for (intptr_t i = 0; i < source.size() - 1; ++i) {
    output << "}\n";
  }
  output << '\n';
}

// TODO: platform-specific
const char* kIntTypes[4] = {"char", "short", "int", "long"};

void GenerateTypeRef(const AType* type, std::ostream& output) {
  if (type->IsNamed()) {
    GenerateSourceIdRef(type->AsNamed()->name, output);
  } else if (type->IsInt()) {
    auto* T = type->AsInt();
    if (T->conv == AIntType::Conversion::kChar) {
      // TODO: maybe use wchar_t?
    }
    if (T->conv == AIntType::Conversion::kUnsigned) {
      output << "unsigned ";
    }
    // TODO: validate?
    auto log = static_cast<intptr_t>(std::log2(T->len.bytes()));
    output << kIntTypes[log];
  } else if (type->IsArray()) {
    output << "std::array<";
    GenerateTypeRef(type->AsArray()->el, output);
    output << ", " << type->AsArray()->length << ">";
  } else if (type->IsList()) {
    auto* L = type->AsList();

    output << "pj::ArrayView<";
    GenerateTypeRef(L->el, output);
    output << "," << L->min_len << "," << L->max_len << ">";
  } else {
    UNREACHABLE();
  }
}

void PrintNamespacedName(const SourceId& name, std::ostream& output) {
  for (auto& p : name) output << "::" << p;
}

void BuildTypeGeneratorStmt(const AType* type, std::ostream& output) {
  if (!type) {
    // Normalize null types to the unit type for the rest of ProtoJIT.
    // These are represented differently from an empty struct in the
    // source generator because in C++ an empty struct has size 8 bits,
    // not 0 bits.
    output << "type = scope->CUnit();";
    return;
  }
  if (type->IsInt()) {
    auto* itype = type->AsInt();
    output << "type = new (scope) CIntType(";
    // First arg: abstract type
    output << "new (scope) AIntType(Bits(" << itype->len.bits() << "),";
    switch (itype->conv) {
      case AIntType::Conversion::kSigned:
        output << "AIntType::Conversion::kSigned";
        break;
      case AIntType::Conversion::kUnsigned:
        output << "AIntType::Conversion::kUnsigned";
        break;
      case AIntType::Conversion::kChar:
        output << "AIntType::Conversion::kChar";
        break;
    }
    output << "),\n";

    assert(itype->len.bytes() <= kMaxCppIntSize);

    // Second arg: alignment
    output << "Bytes(" << itype->len.bytes() << "),\n";

    // Thid arg: total size
    output << "Bytes(" << itype->len.bytes() << "));\n";
  } else if (type->IsArray()) {
    BuildTypeGeneratorStmt(type->AsArray()->el, output);
    output << "type = new (scope) CArrayType(\n";
    // First arg: abstract type
    output << "new (scope) AArrayType(type->abs(), " << type->AsArray()->length
           << "),";
    // Second arg: concrete element type
    output << "type,";
    // Third arg: alignment
    output << "Bytes(alignof(";
    GenerateTypeRef(type, output);
    output << ")),\n";
    // Last arg: total size
    output << "Bytes(sizeof(";
    GenerateTypeRef(type, output);
    output << ")));\n";
  } else if (type->IsNamed()) {
    output << "type = BuildConcreteType<";
    PrintNamespacedName(type->AsNamed()->name, output);
    output << ">::Build(scope);";
  } else {
    UNREACHABLE();
  }
}

void GenerateVariant(Scope* scope, const ParsedProtoFile::Decl& decl,
                     const AType* type, std::ostream& output,
                     std::ostream& back) {
  auto* V = type->AsVariant();

  const bool has_value =
      std::any_of(V->terms.begin(), V->terms.end(),
                  [](auto& t) { return t.second != nullptr; });

  GenerateNamespaceBegin(decl.name, output);

  if (!decl.is_enum) {
    output << "struct " << decl.name.back() << " {\n";
    if (has_value) {
      output << "union {\n";
      for (auto& [name, type] : V->terms) {
        if (type) {
          GenerateTypeRef(type, output);
          output << " " << name << ";\n";
        }
      }
      output << "} value;\n";
    }
  }

  // Define an enum class with all options.
  // SAMIR_TODO: use the right type
  if (decl.is_enum) {
    output << "enum class " << decl.name.back() << " : uint8_t {\n";
  } else {
    output << "enum class Kind : uint8_t {\n";
  }

  output << "undef = " << 0 << ",\n";
  for (auto& [name, _] : V->terms) {
    if (auto ti = decl.explicit_tags.find(name);
        ti != decl.explicit_tags.end()) {
      output << name << " = " << static_cast<uint64_t>(ti->second) << ",\n";
    } else {
      output << name << ",\n";
    }
  }

  if (decl.is_enum) {
    output << "\n};\n";
  } else {
    output << "} tag;\n};\n";
  }
  GenerateNamespaceEnd(decl.name, output);

  // Generate a BuildConcreteType specialization for this type.
  back << "namespace pj {\n";
  back << "namespace gen {\n";
  back << "template <>\n"
       << "struct BuildConcreteType<";
  PrintNamespacedName(decl.name, back);
  back << "> {\n";

  back << "static const CType* Build(Scope* scope) {\n";
  back << "const CType* type;\n";
  back << "std::map<std::string, const AType*> aterms;\n";
  back << "std::map<std::string, CVariantType::CTerm> cterms;\n";

  for (auto& [n, t] : V->terms) {
    BuildTypeGeneratorStmt(t, back);

    back << "cterms.emplace(\"" << n << "\", CVariantType::CTerm{\n"
         << ".tag = static_cast<intptr_t>(";
    PrintNamespacedName(decl.name, back);
    if (!decl.is_enum) {
      back << "::Kind";
    }
    back << "::" << n << "), .type = type\n});\n";
    back << "aterms.emplace(\"" << n << "\", type->abs());\n";
  }

  back << "AVariantType* abstract = new(scope) "
          "AVariantType(std::move(aterms));\n";

  back << "type = new (scope) CVariantType(\n";

  // First arg: abs
  back << "abstract,";

  // Second arg: alignment
  back << "Bytes(alignof(";
  PrintNamespacedName(decl.name, back);
  back << ") ),";

  // Third arg: total size
  back << "Bytes(sizeof(";
  PrintNamespacedName(decl.name, back);
  back << ")),";

  // Fourth arg: concrete terms
  back << "std::move(cterms),\n";

  // Fifth arg: term offset
  if (has_value) {
    back << "Bytes(offsetof(";
    PrintNamespacedName(decl.name, back);
    back << ", value)),\n";
  } else {
    back << "Bytes(0),\n";
  }

  // Sixth arg: term size
  if (has_value) {
    back << "Bytes(sizeof(decltype(";
    PrintNamespacedName(decl.name, back);
    back << "::value))),\n";
  } else {
    back << "Bytes(0),\n";
  }

  // Seventh arg: tag offset
  if (decl.is_enum) {
    back << "Bytes(0),\n";
  } else {
    back << "Bytes(offsetof(";
    PrintNamespacedName(decl.name, back);
    back << ", tag)),\n";
  }

  // Last arg: tag size
  if (decl.is_enum) {
    back << "Bytes(sizeof(";
    PrintNamespacedName(decl.name, back);
    back << "))\n";
    back << ");\n";
  } else {
    back << "Bytes(sizeof(decltype(";
    PrintNamespacedName(decl.name, back);
    back << "::tag)))\n";
    back << ");\n";
  }

  back << "return type;\n"
       << "}\n"
       << "};\n"
       << "}  // namespace gen\n"
       << "}  // namespace pj\n"
       << "\n";
}

void GenerateStruct(Scope* scope, const ParsedProtoFile::Decl& decl,
                    std::ostream& output, std::ostream& back) {
  const auto* type = decl.ctype->AsNamed()->named->AsStruct();
  const auto& name = decl.name;

  std::vector<std::pair<std::string, CStructType::CStructField>>
      fields_by_offset;
  for (auto& pair : type->fields) fields_by_offset.emplace_back(pair);
  std::sort(fields_by_offset.begin(), fields_by_offset.end(),
            [&](const auto& x, const auto& y) {
              return x.second.offset < y.second.offset;
            });

  GenerateNamespaceBegin(name, output);

  output << "struct " << name.back() << " {\n";

  for (auto& [n, field] : fields_by_offset) {
    GenerateTypeRef(field.type->abs(), output);
    output << " " << n << ";\n";
  }
  output << "};\n\n";

  for (auto& [n, field] : fields_by_offset) {
    back << "static_assert(sizeof(";
    PrintNamespacedName(name, back);
    back << "::" << n << ") == " << field.type->total_size().bytes() << ");\n";
    back << "static_assert(offsetof(";
    PrintNamespacedName(name, back);
    back << ", " << n << ") == " << field.offset.bytes() << ");\n";
  }

  back << "static_assert(sizeof(";
  PrintNamespacedName(name, back);
  back << ") == " << type->total_size().bytes() << ");\n";

  GenerateNamespaceEnd(name, output);

  // Generate a BuildConcreteType specialization for this type.
  back << "namespace pj {\n";
  back << "namespace gen {\n";
  back << "template <>\n"
       << "struct BuildConcreteType<";
  PrintNamespacedName(name, back);
  back << "> {\n";

  // For each field:
  //   - Build a concrete type for this field:
  //     - either inline for primitive types
  //     - or a call to another specialization of BuildConcreteType
  //       for a composite type
  // Generate an abstract type from created abstract types.
  // Generate a concrete type from created concrete types.

  back << "static const CType* Build(Scope* scope) {\n";
  back << "const CType* type;\n";
  back << "std::map<std::string, const AType*> afields;\n";
  back << "std::map<std::string, CStructType::CStructField> cfields;\n";

  for (auto& [n, field] : type->fields) {
    BuildTypeGeneratorStmt(field.type->abs(), back);

    back << "cfields.emplace(\"" << n << "\", CStructType::CStructField{\n"
         << ".offset = Bytes(offsetof(";
    PrintNamespacedName(name, back);
    back << "," << n << ")),\n"
         << ".type = type\n});\n";
    back << "afields[\"" << n << "\"] = type->abs();\n";
  }

  back << "AType* abstract = new (scope) "
          "AStructType(std::move(afields));\n";

  back << "type = new (scope) CStructType(";

  // First arg: abstract type
  back << "abstract,\n";

  // Second arg: alignment
  back << "Bytes(alignof(";
  PrintNamespacedName(name, back);
  back << ")),\n";

  // Third arg: total size
  back << "Bytes(sizeof(";
  PrintNamespacedName(name, back);
  back << ")),\n";

  // Last arg: fields
  back << "std::move(cfields)\n";

  back << ");\n"
       << "return type;\n"
       << "}\n"
       << "};\n"
       << "}  // namespace gen\n"
       << "}  // namespace pj\n"
       << "\n";
}

void GenerateComposite(Scope* scope, ParsedProtoFile::Decl decl,
                       std::ostream& output, std::ostream& back) {
  auto type = decl.atype;

  type = type->AsNamed()->named;

  if (type->IsStruct()) {
    GenerateStruct(scope, decl, output, back);
  } else if (type->IsVariant()) {
    GenerateVariant(scope, decl, type, output, back);
  } else {
    UNREACHABLE();
  }
};

void GenerateTypedef(const ParsedProtoFile::Decl& decl, std::ostream& output) {
  GenerateNamespaceBegin(decl.name, output);
  output << "using " << decl.name.back();
  output << " = ";
  GenerateTypeRef(decl.atype->AsNamed()->named, output);
  output << ";\n";
  GenerateNamespaceEnd(decl.name, output);
}

void GenerateProtocol(Scope* scope, const SourceId& name, const AType* type,
                      std::ostream& output, std::ostream& back) {
  GenerateNamespaceBegin(name, output);
  output << "struct " << name.back() << "{};\n";
  GenerateNamespaceEnd(name, output);

  back << "namespace pj {\n";
  back << "namespace gen {\n";
  back << "template <>\n"
       << "struct ProtocolHead<";
  PrintNamespacedName(name, back);
  back << "> {\n"
       << "using Head = ";
  GenerateTypeRef(type, back);
  back << ";\n";
  back << "};\n}  // namespace gen\n\n";
  back << "\n}  // namespace pj\n\n";
}

void GenerateHeader(Scope* scope, const ArchDetails& arch,
                    const ParsedProtoFile& file, std::ostream& output) {
  output << "#pragma once\n"
         << "#include<cstddef>\n "
         << "#include \"pj/protojit.hpp\"\n"
         << "#include \"pj/abstract_types.hpp\"\n"
         << "#include \"pj/concrete_types.hpp\"\n"
         << "\n";

  for (auto& import : file.imports) {
    output << "#include \"" << import.c_str() << ".hpp\"\n";
  }

  std::ostringstream back;
  // TODO: setup namespaces outside
  for (auto& decl : file.decls) {
    switch (decl.kind) {
      case ParsedProtoFile::DeclKind::kType:
        GenerateTypedef(decl, output);
        break;
      case ParsedProtoFile::DeclKind::kComposite:
        GenerateComposite(scope, decl, output, back);
        break;
#if 0
      case ParsedProtoFile::DeclKind::kProtocol:
        GenerateProtocol(scope, decl.name, decl.atype, output, back);
        break;
#endif
    }
  }
  output << back.str();
};

}  // namespace pj
