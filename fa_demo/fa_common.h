#ifndef FA_COMMON_H
#define FA_COMMON_H

#include "common.h"

#define NCTX (16 * 1024)
#define BH 16
#define DHEAD 128
#define ISQRTD 0.08838834764831845
#define LOG2_E 1.44269504088896341

#define Q_IDX(I, J) ((I) * DHEAD + (J))
#define K_IDX(I, J) ((I) * DHEAD + (J))
//#define V_IDX(I, J) ((I) * DHEAD + (J))
#define V_IDX(I, J) ((I) + (J) * NCTX)
#define O_IDX(I, J) ((I) * DHEAD + (J))
#define P_IDX(I, J) ((I) * NCTX + (J))

bool checkFAResult(
    const float16_t *res, const float16_t *ref,
    const int firstM = 0,
    const float atol = DEFAULT_ATOL,
    const float rtol = DEFAULT_RTOL);

#endif  // FA_COMMON_H
