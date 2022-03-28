workspace(name = "protojit")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

SKYLIB_VERSION = "1.0.3"

http_archive(
    name = "bazel_skylib",
    sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

git_repository(
    name = "gtest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    remote = "https://github.com/google/googletest",
    shallow_since = "1570114335 -0400",
)

new_git_repository(
    name = "pegtl",
    build_file_content = """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "headers",
    hdrs = glob(["include/tao/**/*.hpp"]),
    strip_include_prefix = "include/tao",
)
""",
    commit = "df0d9ee623d918498846422c013c73240ea3b42c",
    remote = "https://github.com/taocpp/PEGTL.git",
    shallow_since = "1606576150 +0100",
)

git_repository(
    name = "bazel-compdb",
    commit = "aa58494efdf31c3e3525832b3d44d48bb3bc2b0b",
    remote = "https://github.com/grailbio/bazel-compilation-database.git",
    shallow_since = "1605417044 -0800",
)

LLVM_COMMIT = "75e33f71c2dae584b13a7d1186ae0a038ba98838"
LLVM_SHA256 = "9e2ef2fac7525a77220742a3384cafe7a35adc7e5c9750378b2cf25c2d2933f5"

http_archive(
    name = "llvm-project-raw",
    build_file_content = "#empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
)

http_archive(
    name = "llvm-bazel",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-{}/utils/bazel".format(LLVM_COMMIT),
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
)

load("@llvm-bazel//:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

llvm_configure(
    name = "llvm-project",
    src_path = ".",
    src_workspace = "@llvm-project-raw//:WORKSPACE",
    targets = ["X86"],
)

llvm_disable_optional_support_deps()
