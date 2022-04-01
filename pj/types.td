include "mlir/IR/OpBase.td"

def WidthAttr : Attr<CPred<"$_self.isa<::pj::ir::WidthAttr>()">, "width attribute"> {
  let storageType = [{ ::pj::ir::WidthAttr }];
  let returnType = [{ ::pj::Width }];
  let constBuilderCall = "::pj::ir::WidthAttr::get($_builder.getContext(), $0)";
}

def BoundedBufferType :
    Type<CPred<"$_self.isa<::pj::types::BoundedBufferType>()">, "bbuf">;

def RawBufferType :
    Type<CPred<"$_self.isa<::pj::types::RawBufferType>()">, "rbuf">;

def BufferType :
    Type<CPred<"$_self.isa<::pj::types::RawBufferType>() || $_self.isa<::pj::types::BoundedBufferType>()">, "buf">;

def ValueType :
    Type<CPred<"$_self.isa<::pj::types::ValueType>()">, "value">;

def StructType :
    Type<CPred<"$_self.isa<::pj::types::StructType>()">, "struct">;

def VariantType :
    Type<CPred<"$_self.isa<::pj::types::VariantType>()">, "variant">;

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
