#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>

#include <llvm/Target/TargetMachine.h>

#include "defer.hpp"
#include "ir.hpp"
#include "util.hpp"

namespace pj {
using namespace mlir;
using namespace ir2;
using namespace types;

namespace {
// Convert ProtoJit IR types to LLVM.
struct ProtoJitTypeConverter : public mlir::LLVMTypeConverter {
  ProtoJitTypeConverter(mlir::MLIRContext* C) : mlir::LLVMTypeConverter(C) {
    auto char_star_type =
        LLVM::LLVMPointerType::get(IntegerType::get(&getContext(), 8));
    addConversion([=](ValueType type) { return char_star_type; });
    addConversion([=](RawBufferType type) { return char_star_type; });
  }
};

struct LLVMGenPass : public PassWrapper<LLVMGenPass, OperationPass<ModuleOp>> {
  LLVMGenPass(const llvm::TargetMachine* machine) : machine_(machine) {}

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<ir::ProtoJitDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() final;

  mlir::Type int_type(Width width) {
    return IntegerType::get(&getContext(), width.bits(), IntegerType::Signless);
  }

  mlir::Attribute int_attr(Width width, size_t value) {
    return IntegerAttr::get(int_type(width), value);
  }

  mlir::Value buildWordConstant(mlir::Location& L, mlir::OpBuilder& _,
                                size_t value) {
    return buildIntConstant(L, _, Bits(machine_->getPointerSize(0)), value);
  }

  mlir::Value buildIntConstant(mlir::Location& L, mlir::OpBuilder& _,
                               Width width, size_t value) {
    return _.create<ConstantOp>(L, int_type(width), int_attr(width, value));
  }

  mlir::Type char_star_type() {
    return LLVM::LLVMPointerType::get(
        IntegerType::get(&getContext(), Bytes(1).bits()));
  }

  const llvm::TargetMachine* const machine_;
};
}  // namespace

struct ProjectOpLowering : public OpConversionPattern<ProjectOp> {
  ProjectOpLowering(ProtoJitTypeConverter& converter, MLIRContext* C,
                    LLVMGenPass* pass)
      : OpConversionPattern<ProjectOp>(converter, C), pass_(pass) {}

  LogicalResult matchAndRewrite(ProjectOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass_;
};

LogicalResult ProjectOpLowering::matchAndRewrite(
    ProjectOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto source = operands[0];
  auto src_type = op.src().getType();
  auto result = op.getResult().getType();

  if (src_type.isa<types::ValueType>() ||
      src_type.isa<types::RawBufferType>()) {
    // ValueTypes and RawBuffers are represented as 'char*' in LLVM.
    // The output should have the same representation. We can't create
    // a bounded buffer anyway without some reference for the size.
    assert(result.isa<types::ValueType>() ||
           result.isa<types::RawBufferType>());

    auto L = source.getLoc();
    Value val = _.create<LLVM::GEPOp>(
        L, pass_->char_star_type(), source,
        pass_->buildWordConstant(L, _, op.offset().bytes()));
    _.replaceOp(op, val);
  } else {
    // TODO: handle bounded buffer
    assert(false);
  }

  return success();
}

struct TranscodeOpLowering : public OpConversionPattern<TranscodeOp> {
  TranscodeOpLowering(ProtoJitTypeConverter& converter, MLIRContext* C,
                      LLVMGenPass* pass)
      : OpConversionPattern<TranscodeOp>(converter, C), pass_(pass) {}

  LogicalResult matchAndRewrite(TranscodeOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass_;
};

LogicalResult TranscodeOpLowering::matchAndRewrite(
    TranscodeOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter& _) const {
  auto L = op.getLoc();
  auto src_type = op.src().getType();
  auto dst_type = op.dst().getType();

  // Handle conversion of primitives.
  if (src_type.isa<IntType>() && dst_type.isa<IntType>()) {
    auto src = src_type.cast<IntType>(), dst = dst_type.cast<IntType>();

    if (src->width != dst->width && src->sign != dst->sign) {
      // If signness is different, cannot convert size.
      return failure();
    }

    Value src_ptr = _.create<LLVM::BitcastOp>(
        L, LLVM::LLVMPointerType::get(src.toMLIR()), operands[0]);
    Value dst_ptr = _.create<LLVM::BitcastOp>(
        L, LLVM::LLVMPointerType::get(dst.toMLIR()), operands[1]);

    Value val = _.create<LLVM::LoadOp>(L, src_ptr, src->alignment.bytes());

    // Zero, sign extend, or truncate if necessary.
    if (src->width < dst->width) {
      if (src->sign == Int::Sign::kSigned) {
        val = _.create<LLVM::SExtOp>(L, dst.toMLIR(), val);
      } else {
        val = _.create<LLVM::ZExtOp>(L, dst.toMLIR(), val);
      }
    } else {
      val = _.create<LLVM::TruncOp>(L, dst.toMLIR(), val);
    }

    _.create<LLVM::StoreOp>(op.getLoc(), val, dst_ptr, dst->alignment.bytes());

    _.eraseOp(op);
    return success();
  }

  // Other forms of transcode are not legal at this point.
  return failure();
}

struct TagOpLowering : public OpConversionPattern<TagOp> {
  TagOpLowering(ProtoJitTypeConverter& converter, MLIRContext* C,
                LLVMGenPass* pass)
      : OpConversionPattern<TagOp>(converter, C), pass_(pass) {}

  LogicalResult matchAndRewrite(TagOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass_;
};

LogicalResult TagOpLowering::matchAndRewrite(
    TagOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto L = op.getLoc();

  auto var_type = op.dst().getType().cast<VariantType>();

  auto store_ptr = _.create<LLVM::BitcastOp>(
      L, LLVM::LLVMPointerType::get(pass_->int_type(var_type.tag_width())),
      operands[0]);

  auto tag_cst = pass_->buildIntConstant(L, _, var_type.tag_width(), op.tag());

  _.create<LLVM::StoreOp>(op.getLoc(), tag_cst, store_ptr);

  _.eraseOp(op);
  return success();
}

struct MatchOpLowering : public OpConversionPattern<MatchOp> {
  MatchOpLowering(ProtoJitTypeConverter& converter, MLIRContext* C,
                  LLVMGenPass* pass)
      : OpConversionPattern<MatchOp>(converter, C), pass_(pass) {}

  LogicalResult matchAndRewrite(MatchOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& _) const final;

  LLVMGenPass* const pass_;
};

LogicalResult MatchOpLowering::matchAndRewrite(
    MatchOp op, ArrayRef<Value> operands, ConversionPatternRewriter& _) const {
  auto L = op.getLoc();

  // Load the tag so we can switch on it.
  auto var_type = op.var().getType().cast<VariantType>();

  auto tag_ptr = _.create<LLVM::GEPOp>(
      L, pass_->char_star_type(), operands[0],
      pass_->buildWordConstant(L, _, var_type.tag_offset().bytes()));

  auto load_ptr = _.create<LLVM::BitcastOp>(
      L, LLVM::LLVMPointerType::get(pass_->int_type(var_type.tag_width())),
      tag_ptr);

  Value tag_val = _.create<LLVM::LoadOp>(op.getLoc(), load_ptr);

  // TODO: llvm case expects 32-bit ints, we allow up to 64-bit tags
  tag_val =
      _.create<LLVM::ZExtOp>(op.getLoc(), pass_->int_type(Bits(32)), tag_val);

  llvm::SmallVector<int32_t, 4> case_values;
  for (auto& term : var_type.terms()) {
    case_values.emplace_back(term.tag);
  }
  std::sort(case_values.begin(), case_values.end());

  // Switch on the tag and dispatch to the appropriate branch.
  _.create<LLVM::SwitchOp>(L,
                           /*value=*/tag_val,
                           /*defaultDestination=*/op.successors()[0],
                           /*defaultOperands=*/ValueRange{},
                           /*caseValues=*/case_values,
                           /*caseDestinations=*/op.successors().drop_front(),
                           /*caseOperands=*/ValueRange{});

  _.eraseOp(op);
  return success();
}

bool isLegalFuncOp(mlir::LLVM::LLVMFuncOp func) {
  // TODO: this is kind of hacky, we're probably not supposed
  // to mutate the op here.
  if (func.getName()[0] == '#') {
    func.linkageAttr(mlir::LLVM::LinkageAttr::get(
        func.getContext(), mlir::LLVM::Linkage::Internal));
  }
  return true;
}

void LLVMGenPass::runOnOperation() {
  auto* C = &getContext();

  ConversionTarget target(*C);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addDynamicallyLegalOp<mlir::LLVM::LLVMFuncOp>(isLegalFuncOp);

  OwningRewritePatternList patterns(C);
  ProtoJitTypeConverter typeConverter(C);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);
  patterns.add<ProjectOpLowering, TranscodeOpLowering, TagOpLowering,
               MatchOpLowering>(typeConverter, C, this);

  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLLVMGenPass(const llvm::TargetMachine* machine) {
  return std::make_unique<LLVMGenPass>(machine);
}

}  // namespace pj
