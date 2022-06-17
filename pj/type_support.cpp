#include "types.hpp"
#include "util.hpp"

namespace pj {
namespace types {

std::string printIndent(llvm::raw_ostream& os, llvm::StringRef indent,
                        bool is_last) {
  os << indent;
  auto child_indent = indent.str();
  if (is_last) {
    os << " └─";
    child_indent += "   ";
  } else {
    os << " ├─";
    child_indent += " │ ";
  }
  return child_indent;
}

void ValueType::printTree(llvm::raw_ostream& os, llvm::StringRef name,
                          llvm::StringRef indent, bool is_last) const {
  auto child_indent = printIndent(os, indent, is_last);

  os << name << " (";
  if (isa<NominalType>()) {
    if (isa<StructType>()) {
      os << "Struct ";
    } else if (isa<InlineVariantType>()) {
      os << "InlineVariant ";
    } else if (isa<OutlineVariantType>()) {
      os << "OutlineVariant ";
    } else {
      UNREACHABLE();
    }
  }
  print(os);
  os << ")\n";

  const auto& children_ = children();

  if (hasDetails()) {
    printIndent(os, child_indent, children_.empty());
    printDetails(os);
    os << "\n";
  }

  for (auto it = children_.begin(); it != children_.end(); ++it) {
    it->first.cast<ValueType>().printTree(os, it->second, child_indent,
                                          std::next(it) == children_.end());
  }
}

bool ValueType::classof(mlir::Type val) {
#define CHECK_TYPE_ID(TYPE) \
  if (val.getTypeID() == TYPE::getTypeID()) return true;

  FOR_EACH_VALUE_TYPE(CHECK_TYPE_ID);

#undef CHECK_TYPE_ID

  return false;
}

size_t ReflectDomain::counter = 0;
size_t WireDomain::counter = 0;

bool DomainAttr::classof(mlir::Attribute attr) {
#define CHECK_TYPE_ID(DOMAIN) \
  if (attr.getTypeID() == DOMAIN##DomainAttr::getTypeID()) return true;

  FOR_EACH_DOMAIN(CHECK_TYPE_ID);

#undef CHECK_TYPE_ID

  return false;
}

PathAttr PathAttr::none(mlir::MLIRContext* ctx) {
  return get(ctx, llvm::ArrayRef<llvm::StringRef>{});
}

PathAttr PathAttr::fromString(mlir::MLIRContext* ctx,
                              llvm::StringRef src_path) {
  if (src_path.size() == 0) {
    return get(ctx, llvm::ArrayRef<llvm::StringRef>{});
  }

  llvm::SmallVector<llvm::StringRef, 4> vec;
  for (size_t pos = 0; pos < src_path.size();) {
    auto end = src_path.find('.', pos + 1);
    if (end == src_path.npos) end = src_path.size();
    vec.push_back({src_path.data() + pos, end - pos});
    pos = end + 1;
  }
  return get(ctx, vec);
}

std::string PathAttr::toString() const {
  std::ostringstream sstr;
  bool first = true;
  for (auto piece : getValue()) {
    if (!first) {
      sstr << ".";
    }
    first = false;
    sstr << std::string_view(piece);
  }
  return sstr.str();
}

PathAttr PathAttr::expand(llvm::StringRef prefix) const {
  llvm::SmallVector<llvm::StringRef, 4> vec;
  vec.push_back(prefix);
  vec.append(getValue().begin(), getValue().end());
  return get(getContext(), vec);
}

bool NominalType::classof(mlir::Type val) {
  return val.getTypeID() == StructType::getTypeID() ||
         VariantType::classof(val);
}

void DispatchHandlerAttr::print(llvm::raw_ostream& os) const {
  path().print(os);
  os << "=" << index();
}

bool ValueType::isEnum() const {
  if (auto var_this = dyn_cast<InlineVariantType>()) {
    return var_this->is_enum;
  }
  return false;
}

}  // namespace types
}  // namespace pj
