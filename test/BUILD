load("@rules_cc//cc:defs.bzl", "cc_library")
load("//:site.bzl", "pj_test")

cc_library(
    name = "test_base",
    srcs = ["harness.hpp"],
    deps = [
        "//pj:runtime",
        "@gtest//:gtest_main",
    ],
)

pj_test(
    name = "test_ints",
    srcs = ["test_ints.cpp"],
    protos = ["ints.pj"],
)

pj_test(
    name = "test_size",
    srcs = ["test_size.cpp"],
    protos = ["size.pj"],
)

pj_test(
    name = "test_structs",
    srcs = ["test_structs.cpp"],
    protos = ["structs.pj"],
)

pj_test(
    name = "test_variants",
    srcs = ["test_variants.cpp"],
    protos = ["variants.pj"],
)

pj_test(
    name = "test_arrays",
    srcs = ["test_arrays.cpp"],
    protos = ["arrays.pj"],
)

pj_test(
    name = "test_spaces",
    srcs = ["test_spaces.cpp"],
    protos = ["spaces.pj"],
)

pj_test(
    name = "test_imports",
    srcs = [
        "ints.pj.hpp",
        "spaces.pj.hpp",
        "test_imports.cpp",
    ],
    proto_deps = [
        "ints.pj",
        "spaces.pj",
    ],
    protos = ["imports.pj"],
)

pj_test(
    name = "test_vectors",
    srcs = ["test_vectors.cpp"],
    protos = ["vectors.pj"],
    size = "medium",
)

pj_test(
    name = "test_protocols",
    srcs = ["test_protocols.cpp"],
    protos = ["protocols.pj", "protocols2.pj"],
)
