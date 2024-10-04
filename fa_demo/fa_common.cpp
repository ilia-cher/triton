#include "fa_common.h"

#include "util.h"

bool checkFAResult(
    const float16_t *res, const float16_t *ref,
    const float atol,
    const float rtol) {
  return checkResultWithStrides<float16_t>(
      NCTX, DHEAD,
      res, ref,
      DHEAD, 1,
      atol, rtol);
}
