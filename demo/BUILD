load("//:site.bzl", "pj_lib", "pj_exe")

package(default_visibility = ["//visibility:public"])

pj_exe(
    name = "demo",
    srcs = ["main.cpp"],
    linkopts = ["-lstdc++fs"],
    deps = ["@llvm-project//llvm:Support"],
    protos = ["adoption.v1.pj", "adoption.v2.pj"],
)
