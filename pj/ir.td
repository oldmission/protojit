include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "types.td"

def ProtoJit_Dialect : Dialect {
  let name = "pj";
  let cppNamespace = "::pj::ir";
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
    ValueType:$src,
    ValueType:$dst,
    BufferType:$buf,
    PathAttr:$path,
    TypedArrayAttrBase<DispatchHandlerAttr, "decode handlers">:$handlers
  );

  let results = (outs BufferType);
}

def TranscodePrimitiveOp : ProtoJit_Op<"transcodeprim"> {
  let arguments = (ins ValueType:$src, ValueType:$dst);
}

def AlignOp : ProtoJit_Op<"align"> {
  let arguments = (ins BufferType:$buf, UI64Attr:$alignment);
  let results = (outs BufferType);
}

def AllocateOp : ProtoJit_Op<"allocate"> {
  let arguments = (ins BufferType:$buf, Index:$len);
  let results = (outs BufferType);
}

// UnitOp generates a placeholder value that will not be used at runtime.
// Lowers to nullptr and is expected to be optimized out by LLVM.
def UnitOp : ProtoJit_Op<"empty"> {
  let results = (outs AnyType);
}

def CopyTagOp : ProtoJit_Op<"copytag"> {
  let arguments = (ins InlineVariantType:$src, InlineVariantType:$dst);
}

// DefaultOp populates a value with a default for that type.
//
// For consistency and to preserve assumptions made by downstream optimizations,
// accepts and returns buffers like TranscodeOp, even though they are never used
// or modified.
def DefaultOp : ProtoJit_Op<"default"> {
  let arguments = (ins
      AnyType:$dst,
      BufferType:$buf,
      TypedArrayAttrBase<DispatchHandlerAttr, "decode handlers">:$handlers
  );
  let results = (outs BufferType);
}

// Calculates the size of the memory allocated within the region.
def SizeOp : ProtoJit_Op<"size", [SingleBlockImplicitTerminator<"::pj::ir::YieldOp">]> {
  let arguments = (ins BoolAttr:$round_up);
  let results = (outs Index:$size);
  let regions = (region SizedRegion<1>:$body);
}

// Extract a reference to some component of a value from a fixed offset. The
// "frozen" attributes indicates that the component is accessible and constant.
def ProjectOp : ProtoJit_Op<"project", [NoSideEffect]> {
  let arguments = (ins
      AnyType:$base, WidthAttr:$offset, DefaultValuedAttr<BoolAttr, "false">:$frozen
  );
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
def MatchOp : ProtoJit_Op<"match", [Terminator]> {
  let arguments = (ins VariantType:$var);
  let successors = (
    successor AnySuccessor:$dflt, VariadicSuccessor<AnySuccessor>:$cases
  );
}

// Retrieve a reference to a value at the given index in an Array.
def ArrayIndexOp : ProtoJit_Op<"arrindex", [NoSideEffect]> {
  let arguments = (ins ArrayType:$arr, Index:$index);
  let results = (outs ValueType);
}

// Reads the stored length of a Vector.
def LengthOp : ProtoJit_Op<"length", [NoSideEffect]> {
  let arguments = (ins VectorType:$vec);
  let results = (outs Index);
}

// Stores the provided length into the length field of a Vector.
def StoreLengthOp : ProtoJit_Op<"storelength"> {
  let arguments = (ins VectorType:$vec, Index:$length);
}

// Stores the provided buffer into the ref field of a Vector.
def StoreRefOp : ProtoJit_Op<"storeref"> {
  let arguments = (ins VectorType:$vec, BufferType:$buf);
}

def Inline : I64EnumAttrCase<"Inline", 0>;
def Partial : I64EnumAttrCase<"Partial", 1>;
def Reference : I64EnumAttrCase<"Reference", 2>;
def Buffer : I64EnumAttrCase<"Buffer", 3>;

def VectorRegion : I64EnumAttr<"VectorRegion", "",
                               [Inline, Partial, Reference, Buffer]>;

// Retrieves a reference to a value at the given index in the given storage
// region of a Vector. The VectorRegion values have the corresponding meanings:
//   Inline: the inline storage of the vector
//   Partial: the partial payload storage of the vector
//   Reference: the region of memory pointed to by the ref field of the vector
//   Buffer: the provided buffer (for writing outline data)
// $buf is only used if $region is set to Buffer
def VectorIndexOp : ProtoJit_Op<"vecindex", [NoSideEffect]> {
  let arguments = (
    ins VectorType:$vec,
    Index:$index,
    VectorRegion:$region,
    BufferType:$buf
  );
  let results = (outs ValueType);
}

// Construct a generic representation of a value into an AnyType dst.
// 'src' must be of "reflectable" type.
//
// Buffer is handled simiarly to 'Transcode.'
def ReflectOp : ProtoJit_Op<"reflect"> {
  let arguments = (ins ValueType:$src, ProtoJitAnyType:$dst);
}

// Terminates execution and causes decoding to return a null buffer without
// calling any callbacks, signalling that insufficient memory was provided.
def ThrowOp : ProtoJit_Op<"throw", [Terminator]> {}

// Register a callback to be invoked when decoding is complete.
def SetCallbackOp : ProtoJit_Op<"setcallback"> {
  let arguments = (ins IndexAttr:$target);
}

// Invoke a callback with the result of decoding.
// Can only appear in the body of DecodeCatchOp.
//
// This is the only op in PJ IR which cannot be re-ordered with other ops.
def InvokeCallbackOp : ProtoJit_Op<"invoke"> {
  let arguments = (
    ins ValueType:$obj, HandlersArrayType:$handlers, UserStateType:$state
  );
}

// Catches buffer exhaustion exceptions thrown during decoding.
// Also serves as the anchor for setting the decoder callback.
//
// Should only have one block in the body, and that block should end
// with a YieldOp.
//
// If an exception is thrown inside, returns a null buffer.
// Otherwise, returns whatever YieldOp returns.
def DecodeCatchOp : ProtoJit_Op<"catch", [SingleBlockImplicitTerminator<"::pj::ir::YieldOp">]> {
  let results = (outs BufferType:$buf);
  let regions = (region SizedRegion<1>:$body);
}

// Yields values and terminates execution within DecodeCatchOp and SizeOp. Must
// be the last operation in each parent op's body.
def YieldOp : ProtoJit_Op<"yield", [Terminator]> {
  let arguments = (ins AnyType:$result);
}

// Mark some area inside an object as being "poisoned"; i.e., the memory
// in that area has undefined bits after this operation.
def PoisonOp : ProtoJit_Op<"poison"> {
  let arguments = (ins AnyType:$result, WidthAttr:$offset, WidthAttr:$width);
}

def AssumeOp : ProtoJit_Op<"assume"> {
  let arguments = (ins I1:$condition);
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
    ins StrAttr:$name,
    TypeAttr:$src,
    TypeAttr:$dst,
    PathAttr:$src_path,
    BoolAttr:$round_up
  );
}
