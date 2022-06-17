load("//:site.bzl", "pj_exe", "pj_lib")

package(default_visibility = ["//visibility:public"])

pj_exe(
    name = "demo",
    srcs = ["main.cpp"],
    linkopts = ["-lstdc++fs"],
    proto = "adoption.pj",
    deps = ["@llvm-project//llvm:Support"],
)
