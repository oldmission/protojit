load("@rules_cc//cc:defs.bzl", "cc_library")
load("//:site.bzl", "pj_lib", "pj_test")

cc_library(
    name = "test_base",
    srcs = ["harness.hpp"],
    deps = [
        "//pj:runtime",
        "//pj:reflect_precompile_obj",
        "@gtest//:gtest_main",
    ],
)

pj_test(
    name = "test_ints",
    srcs = ["test_ints.cpp"],
    proto = "ints.pj",
)

pj_test(
    name = "test_structs",
    srcs = ["test_structs.cpp"],
    proto = "structs.pj",
)

pj_test(
    name = "test_variants",
    size = "medium",
    srcs = ["test_variants.cpp"],
    proto = "variants.pj",
)

pj_test(
    name = "test_arrays",
    srcs = ["test_arrays.cpp"],
    proto = "arrays.pj",
)

pj_test(
    name = "test_spaces",
    srcs = ["test_spaces.cpp"],
    proto = "spaces.pj",
)

pj_test(
    name = "test_imports",
    srcs = [
        "ints.pj.hpp",
        "spaces.pj.hpp",
        "test_imports.cpp",
    ],
    proto = "imports.pj",
    proto_deps = [
        "ints.pj",
        "spaces.pj",
    ],
    deps = [
        "test_ints_lib",
        "test_spaces_lib",
    ],
)

pj_test(
    name = "test_vectors",
    size = "medium",
    srcs = ["test_vectors.cpp"],
    proto = "vectors.pj",
)

pj_test(
    name = "test_sizing",
    srcs = ["test_sizing.cpp"],
    proto = "sizing.pj",
)

pj_test(
    name = "test_hoisting",
    size = "medium",
    srcs = ["test_hoisting.cpp"],
    proto = "hoisting.pj",
)

pj_lib(
    name = "protocols1",
    proto = "protocols.pj",
)

pj_lib(
    name = "protocols2",
    proto = "protocols2.pj",
)

pj_test(
    name = "test_any",
    srcs = ["test_any.cpp"],
    proto = "any.pj",
)

# TODO: restore test
# cc_test(
#     name = "test_protocols",
#     srcs = ["test_protocols.cpp"],
#     deps = [
#         "protocols1",
#         "protocols2",
#         "test_base",
#     ],
# )

pj_test(
    name = "test_portals",
    srcs = [
        "ints.pj.hpp",
        "test_portals.cpp",
    ],
    proto = "portals.pj",
    proto_deps = ["ints.pj"],
    deps = [
        "test_ints_lib",
    ],
)
