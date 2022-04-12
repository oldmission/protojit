include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "types.td"

def ProtoJit_Dialect : Dialect {
  let name = "pj";
  let cppNamespace = "::pj::ir2";
}

class ProtoJit_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ProtoJit_Dialect, mnemonic, traits>;

// TranscodeOp decodes a value from a source to destination type.
//
// Path defines a restriction on what values in the 'src' type can be observed.
// The path may be empty, implying no restriction, or a sequence of the form
// <field name>* <term name>, implying that the variant type identifed by
// following the field names will also hold the tag implied by the term name.
//
// Receives a buffer which it may use to store outline data. The buffer must
// *not* alias with with 'dst'. Returns a new buffer object referencing
// the remaining space in the original buffer. Buffer handling depends on
// whether buf is raw or bounded -- raw buffer accesses are not checked,
// bounded buffer accesses are.
def TranscodeOp : ProtoJit_Op<"transcode"> {
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    BufferType:$buf,
    PathAttr:$path,
    TypedArrayAttrBase<DispatchHandlerAttr, "decode handlers">:$handlers
  );

  let results = (outs BufferType);
}

// UnitOp generates a placeholder value that will not be used at runtime.
// Lowers to nullptr and is expected to be optimized out by LLVM.
def UnitOp : ProtoJit_Op<"empty"> {
  let results = (outs AnyType);
}

// DefaultOp populates a value with a default for that type.
//
// No buffer is accepted because the default value for types with external
// data is empty anyway.
def DefaultOp : ProtoJit_Op<"default"> {
  let arguments = (ins
      AnyType:$dst,
      TypedArrayAttrBase<DispatchHandlerAttr, "decode handlers">:$handlers
  );
}

// Calculate the size that the given value of the provided memory type will
// occupy in the destination type. 'src' or 'dst' may be memory or wire types
// or a protocol.
def SizeOp : ProtoJit_Op<"size"> {
  let arguments = (ins AnyType:$src, AnyType:$dst);
  let results = (outs Index);
}

// Extract a reference to some component of a value from a fixed offset.
// ProjectOp is used to load variable variants until the LowerVariableProjects
// pass, at which point any ProjectOp with a VariableVariant output is
// converted to a ProjectVariableOp.
def ProjectOp : ProtoJit_Op<"project", [NoSideEffect]> {
  let arguments = (ins AnyType:$src, WidthAttr:$offset);
  let results = (outs AnyType);
}

// Set the tag on a variant, which may be of Fixed or Variable kind.
// Either writes to the tag inside the variant for a FixedVariant, or
// writes to the tag at the beginning of the protocol buffer for a
// VariableVariant.
def TagOp : ProtoJit_Op<"tag"> {
  let arguments = (ins VariantType:$dst, UI64Attr:$tag);
}

// Dispatch on the tag of a variant and jump to the corresponding successor.
// The number of successors should match the number of tag values for the
// incoming variant, and should be listed in the same numerical order.
//
// TODO: rename to match after IR rewrite is complete.
def MatchOp : ProtoJit_Op<"matchtag", [Terminator]> {
  let arguments = (ins VariantType:$var);
  let successors = (
    successor AnySuccessor:$dflt, VariadicSuccessor<AnySuccessor>:$cases
  );
}

// Retrieve a reference to a value at the given index in an Array or Vector.
def IndexOp : ProtoJit_Op<"index", [NoSideEffect]> {
  let arguments = (ins SeqType:$seq, Index:$index);
  let results = (outs ValueType);
}

// Initialize a vector with a given length.
//
// Buffer is handled similarly to 'Transcode'.
def VectorInitOp : ProtoJit_Op<"vecinit"> {
  let arguments = (
    ins VectorType:$vec, AnyUnsignedInteger:$len, BufferType:$buf
  );
  let results = (outs BufferType);
}

// Construct a generic representation of a value into an AnyType dst.
//
// Buffer is handled simiarly to 'Transcode.'
def ReflectOp : ProtoJit_Op<"reflect"> {
  let arguments = (ins ValueType:$src, ProtoJitAnyType:$dst, BufferType:$buf);
  let results = (outs BufferType);
}

// Register a callback to be invoked when decoding is complete.
def SetCallbackOp : ProtoJit_Op<"setcallback"> {
  let arguments = (ins IndexAttr:$target);
}

// Invoke a callback with the result of decoding.
// Can only appear in the body of DecodeCatchOp.
//
// This is the only op in PJ IR which cannot be re-ordered with other ops.
// The arguments may be ValueType and UserState type in ProtoJIT IR, or
// llvm.ptr during LLVM lowering. In either case the arguments are understood
// to be two word-sized pointers arguments to the callee.
def InvokeCallbackOp : ProtoJit_Op<"invoke"> {
  let arguments = (ins AnyType:$arg, AnyType:$state);
}

// Catches buffer exhaustion exceptions thrown during decoding.
// Also serves as the anchor for setting the decoder callback.
//
// Should only have one block in the body, and that block should end
// with a YieldOp.
//
// If an exception is thrown inside, returns a null buffer.
// Otherwise, returns whatever YieldOp returns.
def DecodeCatchOp : ProtoJit_Op<"catch", [SingleBlockImplicitTerminator<"::pj::ir2::YieldOp">]> {
  let results = (outs BufferType:$buf);
  let regions = (region SizedRegion<1>:$body);
}

// Terminate execution inside a DecodeCatchOp, and make it return the
// provided value. Must be the last operation in a DecodeCatchOp's body.
def YieldOp : ProtoJit_Op<"yield", [Terminator]> {
  let arguments = (ins AnyType:$result);
}

////////////////////////////////////////////////////////////////
// These operations represent functions whose compilation was requested by
// the user. They merely serve as markers which get elaborated during the
// first compilation pass.

def EncodeFunctionOp : ProtoJit_Op<"encodefunc"> {
  let arguments = (
    ins StrAttr:$name, TypeAttr:$src, TypeAttr:$dst, PathAttr:$src_path
  );
}

def DecodeFunctionOp : ProtoJit_Op<"decodefunc"> {
  let arguments = (
    ins StrAttr:$name, TypeAttr:$src, TypeAttr:$dst,
    TypedArrayAttrBase<DispatchHandlerAttr, "decode handlers">:$handlers
  );
}

def SizeFunctionOp : ProtoJit_Op<"sizefunc"> {
  let arguments = (
    ins StrAttr:$name, TypeAttr:$src, OptionalAttr<PathAttr>:$src_path,
    TypeAttr:$dst
  );
}
