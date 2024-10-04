#ifndef FA_COMMON_H
#define FA_COMMON_H

#include "common.h"

#define NCTX (16 * 1024)
#define BH 16
#define DHEAD 128
#define ISQRTD 0.08838834764831845

#define Q_IDX(I, J) ((I) * DHEAD + (J))
#define K_IDX(I, J) ((I) * DHEAD + (J))
#define V_IDX(I, J) ((I) * DHEAD + (J))
#define O_IDX(I, J) ((I) * DHEAD + (J))
#define P_IDX(I, J) ((I) * NCTX + (J))

bool checkFAResult(
    const float16_t *res, const float16_t *ref,
    const float atol = DEFAULT_ATOL,
    const float rtol = DEFAULT_RTOL);

#endif  // FA_COMMON_H
