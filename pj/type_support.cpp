#include "types.hpp"
#include "util.hpp"

namespace pj {
namespace types {

bool ValueType::classof(mlir::Type val) {
  return val.getTypeID() == IntType::getTypeID() ||
         val.getTypeID() == StructType::getTypeID() ||
         val.getTypeID() == InlineVariantType::getTypeID() ||
         val.getTypeID() == OutlineVariantType::getTypeID() ||
         val.getTypeID() == ArrayType::getTypeID() ||
         val.getTypeID() == VectorType::getTypeID() ||
         val.getTypeID() == AnyType::getTypeID() ||
         val.getTypeID() == ProtocolType::getTypeID();
}

void PathAttr::print(llvm::raw_ostream& os) const {
  bool first = true;
  for (auto& part : getValue()) {
    if (!first) {
      os << ".";
    }
    os << part;
    first = false;
  }
}

PathAttr PathAttr::none(mlir::MLIRContext* C) {
  return get(C, llvm::ArrayRef<llvm::StringRef>{});
}

PathAttr PathAttr::fromString(mlir::MLIRContext* C, llvm::StringRef src_path) {
  if (src_path.size() == 0) {
    return get(C, llvm::ArrayRef<llvm::StringRef>{});
  }

  llvm::SmallVector<llvm::StringRef, 2> vec;
  for (size_t pos = 0; pos < src_path.size();) {
    auto end = src_path.find('.', pos + 1);
    if (end == src_path.npos) end = src_path.size();
    vec.push_back({src_path.data() + pos, end - pos});
    pos = end + 1;
  }
  return get(C, vec);
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

bool NominalType::classof(mlir::Type val) {
  return val.getTypeID() == StructType::getTypeID() ||
         VariantType::classof(val);
}

void DispatchHandlerAttr::print(llvm::raw_ostream& os) const {
  path().print(os);
  os << reinterpret_cast<uintptr_t>(address());
}

}  // namespace types
}  // namespace pj
