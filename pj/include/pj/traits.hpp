#pragma once

namespace pj {

template <typename T>
struct wrapped_type {
  using type = T;
};

namespace gen {

template <typename T>
struct BuildPJType;

template <typename T>
struct BuildPJProtocol;

}  // namespace gen

}  // namespace pj
