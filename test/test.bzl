load("@rules_cc//cc:defs.bzl", "cc_test")
load("//:site.bzl", "pj_library")

def pj_test(name, srcs, protos, deps = []):
    pj_library(
        name = name + "_pj",
        srcs = protos,
        deps = deps,
    )
    cc_test(
        name = name,
        srcs = srcs,
        deps = deps + [name + "_pj", "//test:test_base"],
    )
