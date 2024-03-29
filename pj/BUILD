load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(default_visibility = ["//visibility:public"])

# A lot of code in LLVM is not clean under GCC's warnings. These flags allow us to
# enable -Werror under GCC for just the protojit source during testing, but leave it
# off for LLVM.
string_flag(
    name = "build_type",
    build_setting_default = "release",
)

config_setting(
    name = "release",
    flag_values = {
        ":build_type": "release",
    },
)

config_setting(
    name = "test",
    flag_values = {
        ":build_type": "test",
    },
)

COPTS = select({
    ":release": [],
    ":test": ["-Werror"],
})

td_library(
    name = "TypesTdFiles",
    srcs = ["types.td"],
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

gentbl_cc_library(
    name = "ProtoJITIROpsIncGen",
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir.hpp.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir.cpp.inc",
        ),
        (
            ["-gen-enum-decls"],
            "enums.hpp.inc",
        ),
        (
            ["-gen-enum-defs"],
            "enums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir.td",
    deps = [
        ":TypesTdFiles",
        "@llvm-project//mlir:LLVMOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "LLVMExtraIncGen",
    strip_include_prefix = "",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "llvm_extra.hpp.inc",
        ),
        (
            ["-gen-op-defs"],
            "llvm_extra.cpp.inc",
        ),
        (
            ["-gen-llvmir-conversions"],
            "llvm_extra_conversions.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "llvm_extra.td",
    deps = [
        "@llvm-project//mlir:LLVMOpsTdFiles",
    ],
)

cc_library(
    name = "core_noany",
    hdrs = [
        "include/pj/arch_base.hpp",
        "include/pj/array.hpp",
        "include/pj/float.hpp",
        "include/pj/integer.hpp",
        "include/pj/offset_span.hpp",
        "include/pj/runtime.h",
        "include/pj/runtime.hpp",
        "include/pj/span.hpp",
        "include/pj/traits.hpp",
        "include/pj/unit.hpp",
        "include/pj/util.hpp",
    ],
    includes = ["include"],
    copts = COPTS,
)

cc_library(
    name = "compiler_support",
    srcs = [
        "enums.cpp.inc",
        "enums.hpp.inc",
        "ir.cpp",
        "ir.cpp.inc",
        "ir.hpp.inc",
        "type_support.cpp",
        "types.cpp",
    ],
    hdrs = [
        "arch.hpp",
        "array_ref.hpp",
        "defer.hpp",
        "exceptions.hpp",
        "ir.hpp",
        "type_domain.hpp",
        "type_support.hpp",
        "types.hpp",
    ],
    deps = [
        ":core_noany",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
    copts = COPTS,
)

cc_library(
    name = "plan",
    srcs = [
        "convert_internal.cpp",
        "convert_internal.hpp",
        "plan.cpp",
        "variant_outlining.cpp",
        "variant_outlining.hpp",
        "vector_hoisting.cpp",
        "vector_hoisting.hpp",
        "wire_layout.cpp",
        "wire_layout.hpp",
    ],
    hdrs = [
        "plan.hpp",
    ],
    linkopts = ["-lstdc++"],
    deps = [":compiler_support"],
    copts = COPTS,
)

cc_binary(
    name = "pjc",
    srcs = [
        "parse.cpp",
        "pjc.cpp",
        "protogen.hpp",
        "sourcegen.cpp",
        "sourcegen.hpp",
        "validate.cpp",
        "validate.hpp",
    ],
    linkopts = ["-lstdc++fs"],
    deps = [
        ":compiler_support",
        ":plan",
        "@pegtl//:headers",
    ],
    copts = COPTS,
)

genrule(
    name = "reflect_pj",
    srcs = ["reflect.pj"],
    outs = ["include/pj/reflect.pj.hpp"],
    cmd = "$(location //pj:pjc) $(location reflect.pj) --hpp $@",
    tools = [":pjc"],
)

cc_library(
    name = "core",
    hdrs = [
        "core_noany",
        "include/pj/any.hpp",
        "include/pj/reflect.pj.hpp",
    ],
    deps = [":core_noany"],
    copts = COPTS,
)

cc_library(
    name = "reflect",
    srcs = ["reflect.cpp"],
    hdrs = ["reflect.hpp"],
    linkopts = ["-lstdc++fs"],
    deps = [
        ":compiler_support",
        ":core",
    ],
    copts = COPTS,
)

cc_library(
    name = "compiler",
    srcs = [
        "copy_coalescing.cpp",
        "copy_extending.cpp",
        "gdb.cpp",
        "generate_ir.cpp",
        "llvm_extra.cpp",
        "llvm_extra.cpp.inc",
        "llvm_extra.hpp.inc",
        "llvm_extra_conversions.cpp.inc",
        "llvm_gen2.cpp",
        "llvm_utils.cpp",
        "llvm_utils.hpp",
        "side_effect_analysis.cpp",
        "side_effect_analysis.hpp",
        "sizing.cpp",
    ],
    hdrs = [
        "llvm_extra.hpp",
        "passes.hpp",
    ],
    linkopts = ["-lstdc++"],
    deps = [
        ":compiler_support",
        ":core",
        ":reflect",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardToLLVM",
        "@llvm-project//mlir:Transforms",
    ],
    copts = COPTS,
)

# Contains stub implementations of API methods getProtoSize, encodeProto, and
# decodeProto. They are weak symbols, and are overriden by the ones provided by
# schema_precompile when it is linked in. This library containing only the weak
# symbols is used to bootstrap the generation of the precompiled methods
# (:schema_precompile_gen).
cc_library(
    name = "runtime_bootstrap",
    srcs = [
        "context.cpp",
        "context.hpp",
        "portal.hpp",
        "runtime.cpp",
        "schema/precompile.hpp",
        "schema/reflect.pj.v0_1.hpp",
        "schema/versions.hpp",
    ],
    linkopts = ["-lstdc++"],
    deps = [
        ":compiler",
        ":plan",
        ":reflect",
    ],
    alwayslink = 1,
    copts = COPTS,
)

cc_binary(
    name = "schema_precompile_gen",
    srcs = [
        "schema/precompile.cpp",
        "schema/reflect.pj.v0_1.hpp",
        "schema/versions.hpp",
    ],
    deps = [":runtime_bootstrap"],
    linkstatic = False,
    copts = COPTS,
)

genrule(
    name = "schema_precompile_obj",
    srcs = [],
    outs = ["schema_precompile.a"],
    cmd = "$(location schema_precompile_gen) $@",
    tools = ["schema_precompile_gen"],
)

cc_import(
    name = "schema_precompile",
    static_library = "schema_precompile.a",
    alwayslink = 1,
)

cc_library(
    name = "runtime",
    linkopts = ["-lstdc++"],
    deps = [
        ":runtime_bootstrap",
        ":schema_precompile",
    ],
    copts = COPTS,
)

cc_shared_library(
    name = "protojit",
    roots = [":runtime"],
    # Yes... all transitive dependencies have to be listed here :(
    static_deps = [
        ":core",
        ":core_noany",
        ":compiler",
        ":compiler_support",
        ":plan",
        ":reflect",
        ":runtime_bootstrap",
        ":schema_precompile",
        "@llvm_terminfo//:terminfo",
        "@llvm_zlib//:zlib",
        "@llvm-project//llvm:AggressiveInstCombine",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:asm_printer_defs",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:attributes_gen",
        "@llvm-project//llvm:BinaryFormat",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitstreamReader",
        "@llvm-project//llvm:BitstreamWriter",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CFGuard",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:common_transforms",
        "@llvm-project//llvm:config",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Coroutines",
        "@llvm-project//llvm:DebugInfo",
        "@llvm-project//llvm:DebugInfoCodeView",
        "@llvm-project//llvm:DebugInfoDWARF",
        "@llvm-project//llvm:DebugInfoMSF",
        "@llvm-project//llvm:Demangle",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:FrontendOpenMP",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:InstCombineTableGen",
        "@llvm-project//llvm:Instrumentation",
        "@llvm-project//llvm:intrinsic_enums_gen",
        "@llvm-project//llvm:intrinsics_impl_gen",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:MCDisassembler",
        "@llvm-project//llvm:MCParser",
        "@llvm-project//llvm:MLPolicies",
        "@llvm-project//llvm:ObjCARC",
        "@llvm-project//llvm:Object",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcShared",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:pass_registry_def",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:ProfileData",
        "@llvm-project//llvm:Remarks",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TextAPI",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:Vectorize",
        "@llvm-project//llvm:X86CodeGen",
        "@llvm-project//llvm:X86CommonTableGen",
        "@llvm-project//llvm:X86Disassembler",
        "@llvm-project//llvm:X86DisassemblerInternalHeaders",
        "@llvm-project//llvm:X86Info",
        "@llvm-project//llvm:X86UtilsAndDesc",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineMemoryOpInterfacesIncGen",
        "@llvm-project//mlir:AffineOpsIncGen",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:AMX",
        "@llvm-project//mlir:AMXConversionIncGen",
        "@llvm-project//mlir:AMXIncGen",
        "@llvm-project//mlir:AMXToLLVMIRTranslation",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArmNeon",
        "@llvm-project//mlir:ArmNeonConversionIncGen",
        "@llvm-project//mlir:ArmNeonIncGen",
        "@llvm-project//mlir:ArmNeonToLLVMIRTranslation",
        "@llvm-project//mlir:ArmSVE",
        "@llvm-project//mlir:ArmSVEConversionIncGen",
        "@llvm-project//mlir:ArmSVEIncGen",
        "@llvm-project//mlir:ArmSVEToLLVMIRTranslation",
        "@llvm-project//mlir:BuiltinAttributesIncGen",
        "@llvm-project//mlir:BuiltinDialectIncGen",
        "@llvm-project//mlir:BuiltinLocationAttributesIncGen",
        "@llvm-project//mlir:BuiltinOpsIncGen",
        "@llvm-project//mlir:BuiltinTypeInterfacesIncGen",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:BuiltinTypesIncGen",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:CallOpInterfacesIncGen",
        "@llvm-project//mlir:CastOpInterfaces",
        "@llvm-project//mlir:CastOpInterfacesIncGen",
        "@llvm-project//mlir:CommonFolders",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:ControlFlowInterfacesIncGen",
        "@llvm-project//mlir:ConversionPassIncGen",
        "@llvm-project//mlir:CopyOpInterface",
        "@llvm-project//mlir:CopyOpInterfaceIncGen",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:DataLayoutInterfacesIncGen",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InferTypeOpInterfaceIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgInterfacesIncGen",
        "@llvm-project//mlir:LinalgNamedStructuredOpsTcIncGen",
        "@llvm-project//mlir:LinalgNamedStructuredOpsYamlIncGen",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgOpsIncGen",
        "@llvm-project//mlir:LinalgStructuredOpsIncGen",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMConversionIncGen",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMDialectAttributesIncGen",
        "@llvm-project//mlir:LLVMDialectInterfaceIncGen",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMOpsIncGen",
        "@llvm-project//mlir:LLVMPassIncGen",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:LoopLikeInterfaceIncGen",
        "@llvm-project//mlir:MathBaseIncGen",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathOpsIncGen",
        "@llvm-project//mlir:MemRefBaseIncGen",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefOpsIncGen",
        "@llvm-project//mlir:NVVMConversionIncGen",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMOpsIncGen",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:OpAsmInterfaceIncGen",
        "@llvm-project//mlir:OpenACCDialect",
        "@llvm-project//mlir:OpenACCOpsIncGen",
        "@llvm-project//mlir:OpenACCToLLVM",
        "@llvm-project//mlir:OpenACCToLLVMIRTranslation",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:OpenMPOpsIncGen",
        "@llvm-project//mlir:OpenMPToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:ParserTokenKinds",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:PDLInterpOpsIncGen",
        "@llvm-project//mlir:PDLOpsIncGen",
        "@llvm-project//mlir:PDLToPDLInterp",
        "@llvm-project//mlir:PDLTypesIncGen",
        "@llvm-project//mlir:RegionKindInterfaceIncGen",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:ROCDLConversionIncGen",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:ROCDLOpsIncGen",
        "@llvm-project//mlir:ROCDLToLLVMIRTranslation",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:SCFIncGen",
        "@llvm-project//mlir:SCFPassIncGen",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SideEffectInterfacesIncGen",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsIncGen",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:StandardOpsTransformsPassIncGen",
        "@llvm-project//mlir:StandardToLLVM",
        "@llvm-project//mlir:SubElementInterfacesIncGen",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:SymbolInterfacesIncGen",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorEncodingIncGen",
        "@llvm-project//mlir:TensorOpsIncGen",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TransformsPassIncGen",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Translation",
        "@llvm-project//mlir:VectorInterfaces",
        "@llvm-project//mlir:VectorInterfacesIncGen",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:VectorOpsIncGen",
        "@llvm-project//mlir:ViewLikeInterface",
        "@llvm-project//mlir:ViewLikeInterfaceIncGen",
        "@llvm-project//mlir:X86Vector",
        "@llvm-project//mlir:X86VectorConversionIncGen",
        "@llvm-project//mlir:X86VectorIncGen",
        "@llvm-project//mlir:X86VectorToLLVMIRTranslation",
    ],
)

cc_binary(
    name = "pjcat",
    srcs = ["pjcat.cpp"],
    linkopts = ["-lstdc++fs"],
    dynamic_deps = [":protojit"],
    deps = [
        "@llvm-project//llvm:Support",
        ":core"
    ],
)
