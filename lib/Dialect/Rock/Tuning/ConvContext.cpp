#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "triton/Dialect/Rock/IR/Rock.h"
#include "triton/Dialect/Rock/Tuning/ConvContext.h"

using namespace mlir;
using namespace mlir::rock;

static int getOptionalIntAttribute(Operation *op, StringRef attrName,
                                   int defaultValue) {
  if (op->hasAttrOfType<IntegerAttr>(attrName)) {
    return op->getAttrOfType<IntegerAttr>(attrName).getInt();
  }
  return defaultValue;
}

static void
populateDimIndexAndSize(const ArrayAttr &layoutAttr,
                        const ArrayRef<int64_t> &dim,
                        llvm::StringMap<DimIndexAndSize> &dimIndexAndSize) {
  assert(layoutAttr.size() == dim.size());
  size_t dimValSize = layoutAttr.size();
  for (size_t i = 0; i < dimValSize; ++i) {
    auto key = layoutAttr.getValue()[i].cast<StringAttr>().getValue();
    auto value = dim[i];
    dimIndexAndSize[key] = {i, value};
  }
}

ConvolutionDims ConvolutionContext::getConvDims() {
  return ConvolutionDims(dimIndexAndSize["y"].size, dimIndexAndSize["x"].size,
                         dimIndexAndSize["ho"].size, dimIndexAndSize["wo"].size,
                         dimIndexAndSize["hi"].size, dimIndexAndSize["wi"].size,
                         dimIndexAndSize["k"].size, dimIndexAndSize["c"].size,
                         dimIndexAndSize["ni"].size, dimIndexAndSize["g"].size);
}

ConvolutionContext mlir::rock::populateConvContext(Operation *op) {
  ConvOpType opType = convOpTypeFromKernelType(
      cast<RockGemmWrapperInterface>(op).getKernelType());

  assert(isa<RockConvInterface>(op) &&
         "The operation should be a conv-like operation");
  auto convOp = dyn_cast<RockConvInterface>(op);

  // XXX: Do we need these, especially since we're not actually serializing
  // anything to sqlite?
  if (opType == ConvOpType::BwdWeight) {
    assert(op->hasAttrOfType<IntegerAttr>("numCu"));
  }
  auto archVal = op->getAttrOfType<StringAttr>("arch").getValue();
  int numCu = getOptionalIntAttribute(op, "numCu", 64);
  int gemmId = getOptionalIntAttribute(op, "gemmId", 0);

  llvm::StringMap<DimIndexAndSize> dimIndexAndSize;

  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  auto outputLayoutAttr = op->getAttrOfType<ArrayAttr>("output_layout");

  auto strideVal = extractFromI64ArrayAttr(convOp.getStrides());
  auto dilationVal = extractFromI64ArrayAttr(convOp.getDilations());
  auto paddingVal = extractFromI64ArrayAttr(convOp.getPadding());

  populateDimIndexAndSize(
      filterLayoutAttr,
      op->getOperand(0).getType().template cast<MemRefType>().getShape(),
      dimIndexAndSize);
  populateDimIndexAndSize(
      inputLayoutAttr,
      op->getOperand(1).getType().template cast<MemRefType>().getShape(),
      dimIndexAndSize);
  populateDimIndexAndSize(
      outputLayoutAttr,
      op->getOperand(2).getType().template cast<MemRefType>().getShape(),
      dimIndexAndSize);

  Type dataType = cast<RockGemmWrapperInterface>(op).getInputType();

  return {archVal,     numCu,      opType, dimIndexAndSize, strideVal,
          dilationVal, paddingVal, gemmId, dataType};
}
