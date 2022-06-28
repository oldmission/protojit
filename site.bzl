"Site definitions for protojit repo"

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain")

ProtoJitInfo = provider(
    "Info needed to depend on a Protocol JIT library.",
    fields = {
        "transitive_sources": "all .pj files which may get sucked in transitively",
        "transitive_includes": "all include paths where .pj files may need to get " +
                               "imported from, relative to the execution root",
    },
)

def _pj_library_impl(ctx):
    # Gather transitive sources.
    srcs = depset(
        direct = ctx.files.srcs,
        transitive = [
            dep[ProtoJitInfo].transitive_sources
            for dep in ctx.attr.deps
        ],
    )

    idir = ctx.attr.include_prefix
    if idir:
        package = ctx.label.package
        if paths.is_absolute(idir):
            idir = idir.lstrip("/")
        else:
            idir = paths.join(package, idir)
        idir = ctx.label.workspace_root + idir

    includes = depset(
        direct = [idir] if idir else [],
        transitive = [
            dep[ProtoJitInfo].transitive_includes
            for dep in ctx.attr.deps
        ],
    )

    # Generate the header file
    outs = []
    files = {}
    for src in ctx.files.srcs:
        hdr_out = ctx.actions.declare_file(src.basename + ".hpp", sibling = src)
        cpp_out = ctx.actions.declare_file(src.basename + ".cpp", sibling = src)

        args = ctx.actions.args()
        args.add(src)
        args.add_all(includes, before_each = "--import-dir")
        args.add("--hpp", hdr_out.path)
        args.add("--cpp", cpp_out.path)
        args.add("--import-dir", ".")

        ctx.actions.run(
            outputs = [hdr_out, cpp_out],
            inputs = srcs,
            executable = ctx.executable._pjc,
            arguments = [args],
            mnemonic = "PJC",
        )

        files[src] = (hdr_out, cpp_out)
        outs += [hdr_out, cpp_out]

    include_dir = paths.join(ctx.genfiles_dir.path, idir)

    direct_cc_info = CcInfo(
        compilation_context = cc_common.create_compilation_context(
            includes = depset(direct = [include_dir]),
            headers = depset(outs),
        ),
    )

    transitive_cc_infos = [ctx.attr._runtime[CcInfo]]
    transitive_cc_infos += [dep[CcInfo] for dep in ctx.attr.deps]
    transitive_cc_infos += [dep[CcInfo] for dep in ctx.attr.cc_deps]

    cc_info = cc_common.merge_cc_infos(
        direct_cc_infos = [direct_cc_info],
        cc_infos = transitive_cc_infos,
    )

    # Do preplanning and precompilation.
    cc_toolchain = find_cc_toolchain(ctx)
    features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
    )

    objs = []
    pic_objs = []
    for src in ctx.files.srcs:
        cpp_file = files[src][1]
        (_, outputs) = cc_common.compile(
            actions = ctx.actions,
            feature_configuration = features,
            cc_toolchain = cc_toolchain,
            srcs = [cpp_file],
            compilation_contexts = [cc_info.compilation_context],
            name = src.basename + ".precompiler",
        )
        link_outputs = cc_common.link(
            actions = ctx.actions,
            feature_configuration = features,
            cc_toolchain = cc_toolchain,
            compilation_outputs = outputs,
            linking_contexts = [cc_info.linking_context],
            name = src.basename + ".precompiler_link",
            link_deps_statically = False,
        )

        obj_out = ctx.actions.declare_file(src.basename + ".o", sibling = src)
        ctx.actions.run(
            outputs = [obj_out],
            arguments = ["o", obj_out.path],
            executable = link_outputs.executable,
            mnemonic = "Precompiling",
        )
        objs.append(obj_out)

        pic_out = ctx.actions.declare_file(src.basename + ".pic.o", sibling = src)
        ctx.actions.run(
            outputs = [pic_out],
            arguments = ["pic", pic_out.path],
            executable = link_outputs.executable,
            mnemonic = "Precompiling",
        )
        pic_objs.append(pic_out)

    comp_outputs = cc_common.create_compilation_outputs(
        objects = depset(direct = objs),
        pic_objects = depset(direct = pic_objs),
    )
    (linking_ctx, _) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name + ".link",
        actions = ctx.actions,
        feature_configuration = features,
        cc_toolchain = cc_toolchain,
        compilation_outputs = comp_outputs,
    )

    return [
        DefaultInfo(
            files = depset(direct = outs + objs + pic_objs),
        ),
        ProtoJitInfo(
            transitive_sources = srcs,
            transitive_includes = includes,
        ),
        cc_common.merge_cc_infos(
            direct_cc_infos = [
                cc_info,
                CcInfo(linking_context = linking_ctx),
            ],
        ),
    ]

pj_library = rule(
    _pj_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "include_prefix": attr.string(),
        "deps": attr.label_list(
            providers = [ProtoJitInfo, CcInfo],
            doc = "protojit dependencies",
        ),
        "cc_deps": attr.label_list(
            providers = [CcInfo],
            doc = "cc dependencies",
        ),
        "_pjc": attr.label(
            default = Label("//pj:pjc"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "_runtime": attr.label(
            default = Label("//pj:runtime"),
            providers = [CcInfo],
        ),
        "_cc_toolchain": attr.label(
            default = Label(
                "@rules_cc//cc:current_cc_toolchain",
            ),
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
