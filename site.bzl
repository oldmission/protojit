"Site definitions for protojit repo"

load("@rules_cc//cc:defs.bzl", "cc_test", "cc_binary")

def pj_test(name, protos, srcs, proto_deps = [], size="small"):
    pj_lib(name, protos, proto_deps)

    cc_test(
        name = name,
        srcs = srcs + [proto + ".hpp" for proto in protos],
        deps = ["test_base"],
        size = size,
    )

def pj_exe(name, srcs, linkopts, deps, protos, proto_deps = []):
    pj_lib(name, protos, proto_deps)

    cc_binary(
        name = name,
        srcs = srcs + [
            "//pj:arch_base.hpp",
            "//pj:portal.hpp",
            "//pj:protojit.hpp",
            "//pj:runtime.h",
            "//pj:runtime.hpp",
            "//pj:reflect.pj.hpp",
        ] + [proto + ".hpp" for proto in protos],
        linkopts = linkopts,
        deps = deps,
        dynamic_deps = ["//pj:protojit"],
    )

def pj_lib(name, protos, proto_deps = []):
    for proto in protos:
        native.genrule(
            name = name + "_" + proto,
            srcs = [proto] + proto_deps,
            outs = [proto + ".hpp"],
            cmd = "$(location //pj:pjc) --import-dir . $(location " + proto + ") >$@ && clang-format -style=LLVM -i $@",
            tools = ["//pj:pjc"],
        )
