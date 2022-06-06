#pragma once

#include <iostream>
#include <sstream>
#include <unordered_set>

#include "arch.hpp"
#include "protogen.hpp"

namespace pj {

class SourceGenerator {
 public:
  SourceGenerator(const ArchDetails& arch) : arch_(arch), counter_(0) {}

  void addTypedef(const SourceId& name, types::ValueType type);
  void addProtocolHead(const SourceId& name, types::ValueType type,
                       types::PathAttr tag_path);

  void addProtocol(const SourceId& name, types::ProtocolType proto);

  // Recursively add subtypes if not added for wire types.
  void addComposite(types::ValueType type, bool is_external = false);

  void generateHeader(std::ostream& output,
                      const std::vector<std::filesystem::path>& imports);

 private:
  enum class Region { kDefs, kBuilders };

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

  void pushDomain(types::TypeDomain domain) {
    auto new_domain = Domain::kUnset;
    if (domain == types::TypeDomain::kHost) {
      new_domain = Domain::kHost;
    } else if (domain == types::TypeDomain::kWire) {
      new_domain = Domain::kWire;
    }
    pushDomain(new_domain);
  }

  void popDomain() {
    if (--depth_ == 0) {
      domain_ = Domain::kUnset;
    }
  }

  bool shouldAdd(types::ValueType type) {
    if (auto nominal = type.dyn_cast_or_null<types::NominalType>()) {
      // Host types must be added manually because they require additional
      // information.
      return nominal.type_domain() == types::TypeDomain::kWire &&
             generated_.find(type.getAsOpaquePointer()) == generated_.end();
    }
    return false;
  }

  std::stringstream& stream() {
    return region_ == Region::kDefs ? defs_ : builders_;
  }

  std::string getUniqueName() { return "_" + std::to_string(counter_++); }

  template <typename Name>
  void beginNamespaceOf(const Name& name) {
    for (size_t i = 0; i < name.size() - 1; ++i) {
      stream() << "namespace " << std::string_view(name[i]) << "{";
    }
  }

  template <typename Name>
  void endNamespaceOf(const Name& name) {
    for (size_t i = 0; i < name.size() - 1; ++i) {
      stream() << "}\n";
    }
    stream() << "\n";
  }

  template <typename Name>
  void printName(const Name& name) {
    for (auto& p : name) stream() << "::" << std::string_view(p);
  }

  void printIntTypeRef(Width width, Sign sign);
  void printTypeRef(types::ValueType type);

  // Generates a variable containing a handle to a runtime type. Assumes that
  // all necessary BuildPJType methods already exist or have already been
  // created.
  std::string createTypeHandle(types::ValueType type);

  std::string buildStringArray(Span<llvm::StringRef> arr);

  void addStructDef(types::StructType type);
  void addStructBuilder(types::StructType type);
  void addStruct(types::StructType type, bool is_external);

  void addVariantDef(types::VariantType type, bool has_value, Width tag_width);
  void addVariantBuilder(types::VariantType type, bool has_value,
                         Width tag_width);
  void addVariant(types::VariantType type, bool is_external);

  Region region_;
  Domain domain_ = Domain::kUnset;
  size_t depth_ = 0;
  std::stringstream defs_;
  std::stringstream builders_;
  std::unordered_set<const void*> generated_;

  ArchDetails arch_;
  size_t counter_;
};

}  // namespace pj
