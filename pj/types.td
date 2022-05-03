include "mlir/IR/OpBase.td"

def WidthAttr : Attr<CPred<"$_self.isa<::pj::types::WidthAttr>()">, "width attribute"> {
  let storageType = [{ ::pj::types::WidthAttr }];
  let returnType = [{ ::pj::Width }];
  let constBuilderCall = "::pj::types::WidthAttr::get($_builder.getContext(), $0)";
}

def BoundedBufferType :
    Type<CPred<"$_self.isa<::pj::types::BoundedBufferType>()">, "bbuf">;

def RawBufferType :
    Type<CPred<"$_self.isa<::pj::types::RawBufferType>()">, "rbuf">;

def DummyBufferType :
    Type<CPred<"$_self.isa<::pj::types::DummyBufferType>()">, "dbuf">;

def BufferType :
    Type<CPred<"$_self.isa<::pj::types::RawBufferType>() ||"
               "$_self.isa<::pj::types::BoundedBufferType>() ||"
               "$_self.isa<::pj::types::DummyBufferType>()">, "buf">;

def ValueType :
    Type<CPred<"$_self.isa<::pj::types::ValueType>()">, "value">;

def IntType :
    Type<CPred<"$_self.isa<::pj::types::IntType>()">, "int">;

def StructType :
    Type<CPred<"$_self.isa<::pj::types::StructType>()">, "struct">;

def VariantType :
    Type<CPred<"$_self.isa<::pj::types::VariantType>()">, "variant">;

def InlineVariantType :
    Type<CPred<"$_self.isa<::pj::types::InlineVariantType>()">, "inline_variant">;

def ArrayType :
    Type<CPred<"$_self.isa<::pj::types::ArrayType>()">, "vector">;

def VectorType :
    Type<CPred<"$_self.isa<::pj::types::VectorType>()">, "vector">;

def ProtocolType :
    Type<CPred<"$_self.isa<::pj::types::ProtocolType>()">, "protocol">;

def UserStateType :
    Type<CPred<"$_self.isa<pj::ir::UserStateType>()">, "user state">;

def ProtoJitAnyType :
    Type<CPred<"$_self.isa<::pj::types::AnyType>()">, "any">;

def PathAttr : Attr<CPred<"$_self.isa<::pj::types::PathAttr>()">, "path"> {
  let storageType = [{ ::pj::types::PathAttr }];
  // TODO: can we get tablegen to use let us pass a std::string_view for arguments
  // to build() methods but still have it return ArrayRef<StringRef>?
  let returnType = [{ ::pj::types::PathAttr }];
  let constBuilderCall = "::pj::types::PathAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self";
}

def DispatchHandlerAttr : Attr<CPred<"$_self.isa<::pj::types::DispatchHandlerAttr>()">, "dispatchhandler"> {
  let storageType = [{ ::pj::types::DispatchHandlerAttr }];
  let returnType = [{ ::pj::types::DispatchHandlerAttr }];
  let constBuilderCall = "::pj::ir::DispatchHandlerAttr::get($_builder.getContext(), $0)";
}
