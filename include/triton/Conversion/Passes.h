#ifndef TRITON_CONVERSION_PASSES_H
#define TRITON_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/ArithToIndexPass.h"
#include "triton/Conversion/RockToGPU/RockToGPU.h"
#include "triton/Conversion/TritonGPUToLLVM/RockToLLVMPass.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToRockPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
