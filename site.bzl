"Site definitions for protojit repo"

load("@rules_cc//cc:defs.bzl", "cc_test")

def pj_test(name, protos, srcs, proto_deps = []):
    for proto in protos:
        native.genrule(
            name = name + "_" + proto,
            srcs = [proto] + proto_deps,
            outs = [proto + ".hpp"],
            cmd = "$(location //pj:pjc) --import-dir . $(location " + proto + ") >$@ && clang-format -style=LLVM -i $@",
            tools = ["//pj:pjc"],
        )

    cc_test(
        name = name,
        srcs = srcs + [proto + ".hpp" for proto in protos],
        deps = ["test_base"],
        size = "small",
    )
