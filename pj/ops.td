include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def PJ_Dialect : Dialect {
  let name = "pj";
  let cppNamespace = "::pj::ir";
}

def PJ_Type :
    Type<CPred<"$_self.isa<pj::ir::PJType>()">, "PJ types">;

def WidthAttr : Attr<CPred<"$_self.isa<::pj::ir::WidthAttr>()">, "width attribute"> {
  let storageType = [{ ::pj::ir::WidthAttr }];
  let returnType = [{ ::pj::Width }];
  let constBuilderCall = "::pj::ir::WidthAttr::get($_builder.getContext(), $0)";
}

def PJ_Attr : TypeAttrBase<"pj::ir::PJType", "PJ type attribtue">;

def TagType :
    Type<CPred<"$_self.isa<mlir::IntegerType>()">, "Tag type">;

def TagOrIndexType :
    Type<CPred<"$_self.isa<mlir::IntegerType>() || $_self.isa<mlir::IndexType>()">, "Tag type">;

def PJ_IntType :
    Type<CPred<"$_self.isa<pj::ir::PJType>() && $_self.cast<pj::ir::PJType>()->IsInt()">, "PJ int type">;
def PJ_StructType :
    Type<CPred<"$_self.isa<pj::ir::PJType>() && $_self.cast<pj::ir::PJType>()->IsStruct()">, "PJ int type">;
def PJ_VariantType :
    Type<CPred<"$_self.isa<pj::ir::PJType>() && $_self.cast<pj::ir::PJType>()->IsVariant()">, "PJ int type">;
def PJ_ArrayType :
    Type<CPred<"$_self.isa<pj::ir::PJType>() && $_self.cast<pj::ir::PJType>()->IsArray()">, "PJ int type">;
def PJ_ListType :
    Type<CPred<"$_self.isa<pj::ir::PJType>() && $_self.cast<pj::ir::PJType>()->IsList()">, "PJ int type">;

def PJ_UserStateType :
    Type<CPred<"$_self.isa<pj::ir::UserStateType>()">, "PJ user state type">;

class PJ_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PJ_Dialect, mnemonic, traits>;

def XIntOp : PJ_Op<"xint"> {
  let arguments = (ins PJ_IntType:$from, PJ_IntType:$to);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def SIntOp : PJ_Op<"sint"> {
  let arguments = (ins PJ_IntType:$source);
  let results = (outs Index);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def IIntOp : PJ_Op<"iint"> {
  let arguments = (ins PJ_IntType:$to);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def ProjectOp : PJ_Op<"proj", [NoSideEffect]> {
  let arguments = (ins PJ_Type:$value, Index:$byte_offset);
  let printer = [{ return pj::ir::print(p, *this); }];
  let builders = [
    OpBuilder<(
      ins "::mlir::Type":$resultType, "Value":$value, "intptr_t":$offset),
    [{
        build(
          $_builder, $_state, resultType, value,
          ::pj::ir::GetIndexConstant($_state.location, $_builder, offset));
    }]>,
  ];
  let results = (outs PJ_Type);
  let hasCanonicalizer = 1;
}

def RetOp : PJ_Op<"ret", [NoSideEffect, Terminator]> {
  let assemblyFormat = "attr-dict";
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def XStrOp : PJ_Op<"xstr"> {
  let arguments = (ins PJ_StructType:$from, PJ_StructType:$to);
  let printer = [{ return pj::ir::print(p, *this); }];
  let regions = (region AnyRegion:$body);
}

def SStrOp : PJ_Op<"sstr"> {
  let arguments = (ins PJ_StructType:$source);
  let results = (outs Index);
  let printer = [{ return pj::ir::print(p, *this); }];
  let regions = (region AnyRegion:$body);
}

def MatchVariantOp : PJ_Op<"match">  {
  let arguments = (ins PJ_VariantType:$from);
  let results = (outs Variadic<TagType>);
  let printer = [{ return pj::ir::print(p, *this); }];
  let regions = (region AnyRegion:$body);
}

def SListOp : PJ_Op<"slst"> {
  let arguments = (ins PJ_ListType:$source, PJ_Attr:$target);
  let printer = [{ return pj::ir::print(p, *this); }];
  let results = (outs Index);
  let regions = (region AnyRegion:$body);
}

def XArrayOp : PJ_Op<"xary"> {
  let arguments = (ins PJ_ArrayType:$from, PJ_ArrayType:$to);
  let printer = [{ return pj::ir::print(p, *this); }];
  let regions = (region AnyRegion:$xvalue, AnyRegion:$xdefault);
}

def ETagOp : PJ_Op<"etag"> {
  // TODO(3): architecture-dependent
  let arguments = (ins PJ_Type:$to, TagType:$tag, WidthAttr:$offset);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def DTagOp : PJ_Op<"dtag"> {
  let arguments = (ins PJ_Type:$from, WidthAttr:$offset);
  let results = (outs TagType);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def LTagOp : PJ_Op<"ltag"> {
  // Offset in bytes here
  let arguments = (ins PJ_Type:$from, Index:$byte_offset, WidthAttr:$width);
  let results = (outs TagOrIndexType);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def BTagOp : PJ_Op<"btag",
    [DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect, Terminator]> {
  // TODO(3): architecture-dependent
  // TODO(25): limits tag sizes to 64 bits (is that limitation elsewhere?)
  let arguments = (ins TagType:$tag, I64ElementsAttr:$tagOptions);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$successors);
  let assemblyFormat = [{
    `(` $tag `:` type($tag) `)` `:` $tagOptions `->` $successors attr-dict
  }];
  let builders = [
    OpBuilder<(
      ins "Value":$tag, "llvm::ArrayRef<uint64_t>":$tags,
      "BlockRange":$successors),
    [{
        mlir::Type optionsElementType = $_builder.getI64Type();
        auto dataType = mlir::RankedTensorType::get(
            {static_cast<long>(tags.size())}, optionsElementType);
        auto options =
            mlir::DenseIntElementsAttr::get(dataType, tags);
        build($_builder, $_state, tag, options, successors);
    }]>,
  ];
}

def IterOp : PJ_Op<"iter"> {
  let arguments = (
    ins Index:$start, Index:$end,
    Variadic<AnyType>:$induction_variables
  );
  let printer = [{ return pj::ir::print(p, *this); }];
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion:$body);
}

def DispatchOp : PJ_Op<"disp"> {
  let arguments = (ins PJ_Type:$value, PJ_UserStateType:$state, UI64Attr:$target);
  let printer = [{ return pj::ir::print(p, *this); }];
}

def CastOp : PJ_Op<"cast", [NoSideEffect]> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType);
  let printer = [{ return pj::ir::print(p, *this); }];
  let hasFolder = 1;
}

def LRefOp : PJ_Op<"lref", [NoSideEffect]> {
  let arguments = (ins PJ_Type:$base, WidthAttr:$ref_offset, WidthAttr:$ref_size);
  let results = (outs PJ_Type);
  let printer = [{ return pj::ir::print(p, *this); }];
}
