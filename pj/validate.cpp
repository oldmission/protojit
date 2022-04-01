#include "abstract_types.hpp"
#include "concrete_types.hpp"
#include "protocol.hpp"

namespace pj {

void CAnyType::Validate(bool has_tag) const { throw IssueError(17); }

void COutlinedType::Validate(bool has_tag) const { throw IssueError(21); }

void ANamedType::Validate() const { named->Validate(); }

void AIntType::Validate() const {
  if (len.bytes() != 1 && len.bytes() != 2 && len.bytes() != 4 &&
      len.bytes() != 8) {
    throw IssueError(7);
  }
}

void AAnyType::Validate() const {}

void AStructType::Validate() const {
  if (aliases.size()) {
    throw IssueError(6);
  }

  for (auto& [_, type] : fields) {
    type->Validate();
  }

  for (auto& [name, aliases] : aliases) {
    for (auto& alias : aliases) {
      if (fields.count(alias)) {
        std::string error = "Alias " + alias + " is ambiguous.";
        throw CompilerUserError(std::move(error));
      }
    }
  }
}

void AVariantType::Validate() const {
  if (aliases.size()) {
    throw IssueError(6);
  }

  for (auto& [_, type] : terms) {
    type->Validate();
  }

  for (auto& [name, aliases] : aliases) {
    for (auto& alias : aliases) {
      if (terms.count(alias)) {
        std::string error = "Alias " + alias + " is ambiguous.";
        throw CompilerUserError(std::move(error));
      }
    }
  }
}

void AArrayType::Validate() const {
  if (length < 0) {
    throw CompilerUserError("Array length cannot be negative.");
  }

  el->Validate();
}

void AListType::Validate() const {
  if ((min_len != kNone && min_len < 0) || (max_len != kNone && max_len < 0)) {
    throw CompilerUserError("Array length cannot be negative.");
  }

  el->Validate();
}

void AOutlinedType::Validate() const { el->Validate(); }

// TODO(8): check that total size and alignment are OK
// TODO(8): check that abstract type tree lines up

void CStructType::Validate(bool has_tag) const {
  if (fields.size() != abs()->fields.size()) {
    throw CompilerUserError("CStructType incorrect field count");
  }

  for (auto& [name, field] : fields) {
    if (!abs()->fields.count(name)) {
      throw CompilerUserError("No corresponding field.");
    }
    field.type->Validate(true);
  }

  // TODO(8): check that fields don't overlap
  // TODO(8): check that size fits all fields
}

void CVariantType::Validate(bool has_tag) const {
  if (terms.size() != abs()->terms.size()) {
    throw CompilerUserError("CVariantType incorrect tag count");
  }

  auto max_term_size = Bits(0);
  for (auto& [name, tag] : terms) {
    if (!abs()->terms.count(name)) {
      throw CompilerUserError("No corresponding term.");
    }
    tag.type->Validate(true);
    max_term_size = std::max(max_term_size, tag.type->total_size());
  }

  if (tag_offset.IsNone() && tag_offset < max_term_size) {
    throw CompilerUserError("Tag overlaps with data");
  }

  if (tag_offset.IsNone() && !has_tag) {
    throw CompilerUserError("CVariantType cannot exclude tag here");
  }
}

void CArrayType::Validate(bool has_tag) const { el_->Validate(false); }

void CIntType::Validate(bool has_tag) const {}

void CListType::Validate(bool has_tag) const { el->Validate(false); }

void ProtoSpec::Validate() const { head->ValidateHead(); }

void Protocol::Validate() const {
  head->ValidateHead();
  // TODO: validate head
}

void CNamedType::Validate(bool has_tag) const { named->Validate(has_tag); }

}  // namespace pj
