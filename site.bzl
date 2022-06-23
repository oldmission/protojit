"Site definitions for protojit repo"

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

def pj_test(name, proto, srcs, proto_deps = [], deps = [], size = "small"):
    pj_lib(name + "_lib", proto, proto_deps, deps)

    cc_test(
        name = name,
        srcs = srcs + [proto + ".hpp"],
        deps = ["test_base", name + "_lib"],
        size = size,
    )

def pj_exe(name, srcs, linkopts, deps, proto, proto_deps = []):
    pj_lib(name + "_pj", proto, proto_deps)

    cc_binary(
        name = name,
        srcs = srcs + [
            "//pj:arch_base.hpp",
            "//pj:protojit.hpp",
            "//pj:runtime.h",
            "//pj:runtime.hpp",
            "//pj:reflect.pj.hpp",
        ] + [proto + ".hpp"],
        linkopts = linkopts,
        deps = deps + [name + "_pj"],
        dynamic_deps = ["//pj:protojit"],
    )

def concat(xs):
    return "".join(xs)

def pj_lib(name, proto, proto_deps = [], deps = [], nolib = False):
    cmd = concat([
        "$(location //pj:pjc) --import-dir . $(location ",
        proto,
        ") -hpp $(location ",
        proto,
        ".hpp) --cpp $(location ",
        name,
        "_gen.cpp) && clang-format -style=LLVM -i $(location ",
        proto,
        ".hpp)",
    ])

    native.genrule(
        name = name + "_gen",
        srcs = [proto] + proto_deps,
        outs = [proto + ".hpp", name + "_gen.cpp"],
        cmd = cmd,
        tools = ["//pj:pjc"],
    )

    if not nolib:
      cc_library(name = name + "_hdrs", hdrs = [proto + ".hpp"], deps = deps)

      cc_binary(
          name = name + "_compiler",
          srcs = [name + "_gen.cpp"],
          deps = [name + "_hdrs", "//pj:runtime"],
          # TODO: why is this necessary? If removed, LLVM complains that
          # no targets are registered.
          linkstatic = False,
      )

      native.genrule(
          name = name + "_gen_obj",
          srcs = [],
          outs = [proto + ".o"],
          cmd = "$(location " + name + "_compiler) $@",
          tools = [name + "_compiler"],
      )

    cc_library(
        name = name,
        hdrs = [proto + ".hpp"],
        srcs = [] if nolib else [proto + ".o"],
        deps = deps,
    )
