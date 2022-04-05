#pragma once

#include <pegtl.hpp>

#include "types.hpp"

namespace pj {

// Checks that parsed fields are valid. Throws tao::pegtl::parse_error on
// failure.
void validate(const types::Int&, tao::pegtl::position);
void validate(const types::Struct&, tao::pegtl::position);
void validate(const types::InlineVariant&, tao::pegtl::position);
void validate(const types::Array&, tao::pegtl::position);
void validate(const types::Vector&, tao::pegtl::position);

}  // namespace pj
