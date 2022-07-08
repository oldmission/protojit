#include <unordered_set>

#include <pj/util.hpp>

#include "validate.hpp"

namespace pj {

using namespace types;
using tao::pegtl::parse_error;
using tao::pegtl::position;

void validate(const Int& type, position pos) {
  if (type.width.bits() % 8 != 0 ||
      (type.width.bytes() != 1 && type.width.bytes() != 2 &&
       type.width.bytes() != 4 && type.width.bytes() != 8)) {
    throw parse_error(
        "Non-standard int sizes are not yet supported (allowed sizes: 8, 16, "
        "32, 64)",
        pos);
  }
}

void validate(const Struct& type, position pos) {
  std::unordered_set<std::string> names;
  for (const StructField& f : type.fields) {
    std::string name = f.name.str();
    if (names.find(name) != names.end()) {
      throw parse_error("Duplicate field name '" + name + "' in struct", pos);
    }
    names.insert(name);
  }
}

void validate(const InlineVariant& type, position pos) {
  std::unordered_set<std::string> names;
  std::unordered_set<uint64_t> tags;
  for (const Term& t : type.terms) {
    std::string name = t.name.str();
    if (names.find(name) != names.end()) {
      throw parse_error("Duplicate term name '" + name + "' in variant", pos);
    }
    names.insert(name);

    if (tags.find(t.tag) != tags.end()) {
      throw parse_error("Duplicate tag for term '" + name + "' in variant",
                        pos);
    }
    tags.insert(t.tag);
  }
}

void validate(const Array& type, position pos) {
  if (type.length < 0) {
    throw parse_error("Array length cannot be negative", pos);
  }
}

void validate(const Vector& type, position pos) {
  if (type.min_length < 0) {
    throw parse_error("Minimum length cannot be negative", pos);
  }
  if (type.max_length >= 0 &&
      type.min_length > static_cast<uint64_t>(type.max_length)) {
    throw parse_error("Minimum length greater than maximum length", pos);
  }
}

}  // namespace pj
