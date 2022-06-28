load("//:site.bzl", "pj_library")

package(default_visibility = ["//visibility:public"])

pj_library(
    name = "adoption",
    srcs = ["adoption.pj"],
)

cc_binary(
    name = "demo",
    srcs = ["main.cpp"],
    linkopts = ["-lstdc++fs"],
    deps = [":adoption"],
)
