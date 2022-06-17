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

std::string SourceGenerator::createTypeHandleFromDecl(std::string decl) {
  assert(region_ == Region::kBuilders);
  assert(domain_ == Domain::kHost);

  std::string handle;
  stream() << "const auto* " << (handle = getUniqueName())
           << " = BuildPJType<decltype(" << decl << ")>::build(ctx, domain);\n";
  return handle;
}

std::string SourceGenerator::createTypeHandleFromType(types::ValueType type) {
  assert(region_ == Region::kBuilders);
  auto& os = stream();

  std::string handle = getUniqueName();

  // BuildPJType is always used for NominalTypes under the assumption that pjc
  // is not used for host and wire types at once.
  if (domain_ == Domain::kHost || type.isa<types::NominalType>()) {
    os << "const auto* " << handle << " = BuildPJType<";
    printTypeRef(type);
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

  if (auto U = type.dyn_cast<types::UnitType>()) {
    os << "const PJUnitType* " << handle << " = PJCreateUnitType(ctx);\n";
    return handle;
  }

  if (auto A = type.dyn_cast<types::ArrayType>()) {
    std::string elem_handle = createTypeHandleFromType(A->elem);
    os << "const auto* " << handle << " = PJCreateArrayType(ctx";
    os << ", /*elem=*/" << elem_handle;
    os << ", /*length=*/" << A->length;
    os << ", /*elem_size=*/" << A->elem_size.bits();
    os << ", /*alignment=*/" << A->alignment.bits() << ");\n";
    return handle;
  }

  if (auto V = type.dyn_cast<types::VectorType>()) {
    std::string elem_handle = createTypeHandleFromType(V->elem);
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
    std::string field_handle = [&]() {
      // Ints are excluded because they are generated as regular C++ int types,
      // but their BuildPJType methods must use the Integer class, which
      // contains the Sign enum.
      if (domain_ == Domain::kHost && is_external &&
          !field.type.isa<types::IntType>()) {
        std::vector<llvm::StringRef> field_name{name.begin(), name.end()};
        field_name.emplace_back(field.name);
        return createTypeHandleFromDecl(getNameAsString(field_name));
      } else {
        return createTypeHandleFromType(field.type);
      }
    }();

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
    if (shouldAdd(field.type)) {
      addComposite(field.type);
    }
  }

  if (domain_ == Domain::kHost && !is_external) {
    addStructDef(type, false);
  } else if (domain_ == Domain::kWire) {
    addStructDef(type, true);
  }

  addStructBuilder(type, is_external);
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
      os << "union {\n";
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
    std::string term_handle = [&]() {
      // In addition to excluding ints, same as in structs, Unit types are also
      // excluded because they have no corresponding C++ type.
      if (domain_ == Domain::kHost && is_external &&
          !term.type.isa<types::IntType>() &&
          !term.type.isa<types::UnitType>()) {
        std::vector<llvm::StringRef> union_name{name.begin(), name.end()};
        union_name.emplace_back("value");
        return createTypeHandleFromDecl(getNameAsString(union_name) + "." +
                                        term.name.str());
      } else {
        return createTypeHandleFromType(term.type);
      }
    }();

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
    addVariantDef(type, has_value, tag_width, false);
  } else if (domain_ == Domain::kWire) {
    addVariantDef(type, has_value, tag_width, true);
  }

  addVariantBuilder(type, has_value, tag_width, is_external);
}

void SourceGenerator::addProtocol(const SourceId& name, mlir::Type proto,
                                  types::PathAttr path) {
  cpp_ << "  auto* ";
  printPlanName(cpp_, name);
  cpp_ << " = PJPlanProtocol(ctx, "
          "::pj::gen::BuildPJType<"
       << getNameAsString(proto.cast<types::NominalType>().name())
       << ">::build(ctx, PJGetHostDomain(ctx)), \"";
  path.print(cpp_);
  cpp_ << "\");\n";
}

void SourceGenerator::addPortal(const SourceId& ns,
                                const ParsedProtoFile::Portal& portal) {
  if (!portal.jit_class_name.empty()) {
    addJitClass(portal);
  }
  for (auto& [name, plan] : portal.precomps) {
    addPrecompClass(ns, portal, name, plan);
  }
}

void SourceGenerator::printPlanName(std::ostream& os, const SourceId& plan) {
  os << "plan_" << getNameAsString(plan, "_");
}

std::ostream& SourceGenerator::printDecoderSig(
    std::ostream& os, const std::string& name,
    const ParsedProtoFile::Portal::Decoder& decoder, bool state_template) {
  if (state_template) {
    os << "template <typename S>";
  }
  std::string state = state_template ? "S" : "void";
  os << "BoundedBuffer " << name << "("
     << "const char* msg, " << getNameAsString(decoder.dst)
     << "* result, BoundedBuffer buffer,"
     << "::pj::DecodeHandler<" << getNameAsString(decoder.dst) << ", " << state
     << "> handlers[], " << state << "* state)";
  return os;
}

void SourceGenerator::addPrecompClass(const SourceId& ns,
                                      const ParsedProtoFile::Portal& iface,
                                      const std::string& name,
                                      const SourceId& plan) {
  // 1. Define class in the header with appropriate interface.
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
  // The class declaration goes into `defs_` and the function bodies
  // go into `builders_` to keep the code clean.
  builders_ << "#ifndef PROTOJIT_NO_INTERFACES\n";

  region_ = Region::kDefs;
  beginNamespaceOf(ns, /*is_namespace_name=*/true);

  region_ = Region::kBuilders;
  beginNamespaceOf(ns, /*is_namespace_name=*/true);

  defs_ << "struct " << name << "{\n"
        << "static std::string_view getSchema();\n"
        << name << "() {}\n";

  for (auto& sizer : iface.sizers) {
    auto sizer_name = "user_" + getNameAsString(ns, "_") + "_" + sizer.name;
    defs_ << "size_t " << sizer.name << "(" << getNameAsString(sizer.src)
          << " const* msg);\n";

    region_ = Region::kBuilders;
    builders_ << "extern \"C\" "
              << "size_t " << sizer_name << "(const void*);\n"
              << "size_t  " << name << "::" << sizer.name << "("
              << getNameAsString(sizer.src) << " const* msg) {"
              << "  return " << sizer_name << "(msg);\n"
              << "}\n";
    region_ = Region::kDefs;
  }

  for (auto& encoder : iface.encoders) {
    auto encoder_name = "user_" + getNameAsString(ns, "_") + "_" + encoder.name;
    defs_ << "void " << encoder.name << "(" << getNameAsString(encoder.src)
          << " const*, char* buf);\n";

    region_ = Region::kBuilders;
    builders_ << "extern \"C\" "
              << "void " << encoder_name << "(" << getNameAsString(encoder.src)
              << " const*, char* buf);\n"
              << "void  " << name << "::" << encoder.name << "("
              << getNameAsString(encoder.src) << " const* msg, char* buf) {\n"
              << "  " << encoder_name << "(msg, buf);\n"
              << "}\n";
    region_ = Region::kDefs;
  }

  for (auto& decoder : iface.decoders) {
    printDecoderSig(defs_, decoder.name, decoder, /*state_template=*/true)
        << ";";

    auto decoder_name = "user_" + getNameAsString(ns, "_") + "_" + decoder.name;

    region_ = Region::kBuilders;
    builders_ << "extern \"C\" ";
    printDecoderSig(builders_, decoder_name, decoder, /*state_template=*/false)
        << ";\n";

    printDecoderSig(builders_, name + "::" + decoder.name, decoder,
                    /*state_template=*/true)
        << "{\n";
    builders_ << "return " << decoder_name
              << "(msg, result, buffer, handlers, state);\n}\n";
    region_ = Region::kDefs;
  }

  defs_ << "};\n";  // Terminate the struct.

  region_ = Region::kDefs;
  endNamespaceOf(ns, /*is_namespace_name=*/true);

  region_ = Region::kBuilders;
  endNamespaceOf(ns, /*is_namespace_name=*/true);

  builders_ << "#endif  // PROTOJIT_NO_INTERFACES\n";

  // 2. Generate .cpp file to do precompilation.
  for (auto& sizer : iface.sizers) {
    cpp_ << "  PJAddSizeFunction(ctx, \"" << getNameAsString(ns, "_") << "_"
         << sizer.name << "\", ::pj::gen::BuildPJType<"
         << getNameAsString(sizer.src)
         << ">::build(ctx, PJGetHostDomain(ctx)), ";
    printPlanName(cpp_, plan);
    cpp_ << ", \"";
    sizer.src_path.print(cpp_);
    cpp_ << "\", false);\n";
  }

  for (auto& encoder : iface.encoders) {
    cpp_ << "  PJAddEncodeFunction(ctx, \"" << getNameAsString(ns, "_") << "_"
         << encoder.name << "\", ::pj::gen::BuildPJType<"
         << getNameAsString(encoder.src)
         << ">::build(ctx, PJGetHostDomain(ctx)), ";
    printPlanName(cpp_, plan);
    cpp_ << ", \"";
    encoder.src_path.print(cpp_);
    cpp_ << "\");\n";
  }

  for (auto& decoder : iface.decoders) {
    cpp_ << "  {\n"
         << "    const char** handlers = {\n";
    for (auto& handler : decoder.handlers) {
      cpp_ << "    \"";
      handler.print(cpp_);
      cpp_ << "\",\n";
    }
    cpp_ << "  };\n"
         << "  PJAddDecodeFunction(ctx, \"" << getNameAsString(ns, "_") << "_"
         << decoder.name << "\", ";
    printPlanName(cpp_, plan);
    cpp_ << ", ::pj::gen::BuildPJType<" << getNameAsString(decoder.dst)
         << ">::build(ctx, PJGetHostDomain(ctx)), " << decoder.handlers.size()
         << ", handlers);\n"
         << "}\n";
  }
}

void SourceGenerator::addJitClass(const ParsedProtoFile::Portal& spec) {
  std::cerr << "NYI\n";
  abort();
}

void SourceGenerator::generate(
    const std::filesystem::path& path, std::ostream& output, std::ostream* cpp,
    const std::vector<std::filesystem::path>& imports) {
  output << "#pragma once\n"
         << "#include <cstddef>\n"
         << "#include <string_view>\n"
         << "#include \"pj/protojit.hpp\"\n"
         << "#include \"pj/runtime.h\"\n"
         << "\n";

  for (auto& import : imports) {
    output << "#include \"" << import.c_str() << ".hpp\"\n";
  }

  output << defs_.str();
  output << builders_.str();

  if (cpp) {
    *cpp << "#include <cstdio>\n"
         << "#define PROTOJIT_NO_INTERFACES\n"
         << "#include \"pj/runtime.h\"\n"
         << "#include " << path.filename() << "\n"
         << "int main(int argc, char** argv) {\n"
         << "  if (argc != 2) {\n    fprintf(stderr, \"No output filename "
            "given\\n\""
            ");\n    return 1;\n  }\n"
         << "  PJContext* ctx = PJGetContext();\n"
         << cpp_.str() << "  PJPrecompile(ctx, argv[1]);\n"
         << "  PJFreeContext(ctx);\n"
         << "}\n";
  }
};

}  // namespace pj
