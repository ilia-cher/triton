#include "gemm_common.h"

#include "util.h"

bool checkGemmResult(
    const int M, const int N,
    const float *res, const float *ref,
    const float atol,
    const float rtol) {
  return checkResultWithStrides<float>(
      M, N,
      res, ref,
      CS1, CS2,
      atol, rtol);
}
