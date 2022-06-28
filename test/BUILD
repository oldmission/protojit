load("@rules_cc//cc:defs.bzl", "cc_library")
load("//:site.bzl", "pj_library")
load("test.bzl", "pj_test")

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
    srcs = ["test_imports.cpp"],
    protos = ["imports.pj"],
    deps = [
        "test_ints_pj",
        "test_spaces_pj",
    ],
)

pj_test(
    name = "test_vectors",
    srcs = ["test_vectors.cpp"],
    protos = ["vectors.pj"],
)

pj_test(
    name = "test_sizing",
    srcs = ["test_sizing.cpp"],
    protos = ["sizing.pj"],
)

pj_test(
    name = "test_hoisting",
    srcs = ["test_hoisting.cpp"],
    protos = ["hoisting.pj"],
)

pj_library(
    name = "protocols1",
    srcs = ["protocols.pj"],
)

pj_library(
    name = "protocols2",
    srcs = ["protocols2.pj"],
)

pj_test(
    name = "test_any",
    srcs = ["test_any.cpp"],
    protos = ["any.pj"],
)

# TODO: restore test
# cc_test(
#     name = "test_protocols",
#     srcs = ["test_protocols.cpp"],
#     deps = [
#         ":protocols1",
#         ":protocols2",
#         ":test_base",
#     ],
# )

pj_test(
    name = "test_portals",
    srcs = ["test_portals.cpp"],
    protos = ["portals.pj"],
    deps = [":test_ints_pj"],
)

cc_library(
    name = "external",
    hdrs = ["external.hpp"],
)

pj_library(
    name = "external_pj",
    srcs = ["external.pj"],
    cc_deps = ["external"],
)

cc_test(
    name = "test_external",
    srcs = ["test_external.cpp"],
    deps = [
        ":external_pj",
        ":test_base",
    ],
)
