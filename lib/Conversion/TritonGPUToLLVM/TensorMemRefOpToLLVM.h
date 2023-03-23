#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TENSOR_TO_MEMREF_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TENSOR_TO_MEMREF_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateTensorMemRefOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
