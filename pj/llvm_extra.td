include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def LLVM_LifetimeEndOp : LLVM_ZeroResultIntrOp<"lifetime.end", [1]> {
  let arguments = (ins LLVM_Type:$len, LLVM_Type:$dst);
}
