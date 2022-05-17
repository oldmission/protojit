include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// Copied from LLVMIntrinsicOps.td in a more recent version of MLIR
def LLVM_MemsetOp : LLVM_ZeroResultIntrOp<"memset", [0, 2]> {
  let arguments = (ins
    Arg<LLVM_Type, "", [MemWrite]>:$dst,
    LLVM_Type:$val,
    LLVM_Type:$len,
    LLVM_Type:$isVolatile
  );
}

def LLVM_InvariantStartOp : LLVM_ZeroResultIntrOp<"invariant.start", [1]> {
  let arguments = (ins LLVM_Type:$len, LLVM_Type:$dst);
}
