#include "fa_common.h"

#include "util.h"

bool checkFAResult(
    const float16_t *res, const float16_t *ref,
    const int firstM,
    const float atol,
    const float rtol) {
  return checkResultWithStrides<float16_t>(
      (firstM > 0) ? firstM : (BH * NCTX),
      DHEAD,
      res, ref,
      DHEAD, 1,
      atol, rtol);
}
