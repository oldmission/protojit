#pragma once

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "arch.hpp"
#include "protogen.hpp"

namespace pj {

class SourceGenerator {
 public:
  SourceGenerator(const SourceId& outer_namespace)
      : outer_namespace_(outer_namespace), counter_(0) {}

  void addTypedef(const SourceId& name, types::ValueType type);

  // Recursively add subtypes if not added for wire types.
  void addWireProtocol(const SourceId& name, types::ProtocolType proto);
  void addComposite(types::ValueType type, bool is_external = false);
  void addText(const SourceId& space, const std::string& text);

  void addPortal(const SourceId& ns, const Portal& portal,
                 ParsedProtoFile::Protocol proto);
  void addProtocol(const SourceId& name, ParsedProtoFile::Protocol proto);
  void addPrecompilation(const SourceId& name, const Portal& portal,
                         const SourceId& proto_name,
                         ParsedProtoFile::Protocol proto);

  void generate(const std::filesystem::path& header_path, std::ostream& output,
                std::ostream* cpp,
                const std::vector<std::filesystem::path>& imports);

 private:
  enum class Region { kDefs, kBuilders, kCpp };

  // The type domain of the composite type that is currently being generated.
  // Host types have struct and enum definitions generated, and their build
  // methods are based on the host compiler's layout decisions. Wire types only
  // have build methods generated describing the exact layout of the MLIR types
  // as provided.
  enum class Domain { kUnset, kHost, kWire };

  void pushDomain(Domain domain) {
    assert(domain != Domain::kUnset);
    assert((depth_ == 0 && domain_ == Domain::kUnset) || domain_ == domain);
    domain_ = domain;
    depth_++;
  }

  void pushDomain(types::DomainAttr domain) {
    auto new_domain = Domain::kUnset;
    // .pj files are parsed into the InternalDomain, because the types do not
    // actually correspond to an actual in-memory type. However, since the
    // generated C++ code does describe an in-memory type, it should be
    // generated in HostDomain.
    if (domain.isa<types::InternalDomainAttr>()) {
      new_domain = Domain::kHost;
    } else if (domain.isa<types::WireDomainAttr>()) {
      new_domain = Domain::kWire;
    }
    pushDomain(new_domain);
  }

  void popDomain() {
    if (--depth_ == 0) {
      domain_ = Domain::kUnset;
    }
  }

  // Adds the type as well as all of its subtypes.
  void addType(types::ValueType type);

  std::stringstream& stream() {
    if (region_ == Region::kDefs) return defs_;
    if (region_ == Region::kBuilders) return builders_;
    if (region_ == Region::kCpp) return cpp_;
    UNREACHABLE();
  }

  std::string getUniqueName() { return "_" + std::to_string(counter_++); }

  template <typename Name>
  void beginNamespaceOf(const Name& name, bool is_namespace_name = false) {
    for (std::string_view space : outer_namespace_) {
      stream() << "namespace " << space << "{\n";
    }
    for (intptr_t i = 0; i < name.size() - !is_namespace_name; ++i) {
      stream() << "namespace " << std::string_view(name[i]) << "{\n";
    }
  }

  template <typename Name>
  void endNamespaceOf(const Name& name, bool is_namespace_name = false) {
    for (size_t i = 0;
         i < outer_namespace_.size() + name.size() - !is_namespace_name; ++i) {
      stream() << "}\n";
    }
    stream() << "\n";
  }

  template <typename Name>
  std::string getNameAsString(const Name& name, const char* delim = "::") {
    std::stringstream str;
    for (auto& p : outer_namespace_) str << delim << p;
    for (auto& p : name) str << delim << std::string_view(p);
    return str.str();
  }

  template <typename Name>
  void printName(const Name& name) {
    stream() << getNameAsString(name);
  }

  void printIntTypeRef(Width width, Sign sign, bool wrap = false,
                       std::string decl = {});
  void printTypeRef(types::ValueType type, bool wrap = false,
                    std::string decl = {});
  llvm::StringRef classNameFor(types::ValueType type);
  void printPlanName(std::ostream& os, const SourceId& plan);

  std::ostream& printDecoderSig(std::ostream& os, const std::string& name,
                                const Portal::Decoder& decoder,
                                bool state_template, bool is_declaration);

  // Generates a variable containing a handle to a runtime type generated using
  // the type of the in-memory type, obtained using decltype. Expects a
  // PJContext* in scope with name ctx.
  std::string createTypeHandle(std::string decl, types::ValueType type);

  std::string buildStringArray(ArrayRef<llvm::StringRef> arr);

  void addStructDef(types::StructType type, bool decl_only);
  void addStructBuilder(types::StructType type, bool is_external);
  void addStruct(types::StructType type, bool is_external);

  void addVariantDef(types::VariantType type, bool has_value, Width tag_width,
                     bool decl_only);
  void addVariantBuilder(types::VariantType type, bool has_value,
                         Width tag_width, bool is_external);
  void addVariant(types::VariantType type, bool is_external);
  void addVariantFunctions(types::VariantType type);

  Region region_;
  Domain domain_ = Domain::kUnset;
  size_t depth_ = 0;
  std::stringstream defs_;
  std::stringstream builders_;
  std::stringstream cpp_;
  std::unordered_set<const void*> generated_;
  std::vector<std::string> entrypoints_;
  std::set<SourceId, SourceIdLess> protos_;
  std::set<std::string> base_headers_;

  SourceId outer_namespace_;
  size_t counter_;
};

}  // namespace pj
