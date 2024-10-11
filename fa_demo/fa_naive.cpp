#include "fa_kernels.h"

#include <hip/hip_runtime.h>

__launch_bounds__(FA_NAIVE_NTHREADS, 1)
__global__ void fa_naive(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob) {

  const float16_t *Q = Qb + blockIdx.y * NCTX * DHEAD;
  const float16_t *K = Kb + blockIdx.y * NCTX * DHEAD;
  const float16_t *V = Vb + blockIdx.y * NCTX * DHEAD;
  float16_t *O = Ob + blockIdx.y * NCTX * DHEAD;

  int m = blockIdx.x * FA_NAIVE_BLOCK_M + threadIdx.y * 64 + threadIdx.x;

  //assert(Dhead <= DHEAD);
  float O_fp32[DHEAD] = {0};

  float row_max = -std::numeric_limits<float>::infinity();
  float row_sumexp = 0.0;
  for (int n = 0; n < NCTX; ++n) {
    float tmp = 0.0;
    for (int k = 0; k < DHEAD; ++k) {
      tmp += Q[Q_IDX(m, k)] * K[K_IDX(n, k)];
    }
    tmp *= ISQRTD;

    float alpha = 1.0;
    bool needs_correction = false;
    if (row_max < tmp) {
      alpha = std::expf(row_max - tmp);
      row_max = tmp;
      row_sumexp *= alpha;
      needs_correction = true;
    }

    tmp = std::expf(tmp - row_max);
    row_sumexp += tmp;
    float16_t tmp_fp16 = (float16_t)tmp;
    for (int k = 0; k < DHEAD; ++k) {
      if (needs_correction) {
        float t = O_fp32[k];
        O_fp32[k] *= alpha;
      }

      O_fp32[k] += tmp_fp16 * V[V_IDX(n, k)];
    }
  }

  //
  for (int k = 0; k < DHEAD; ++k) {
    O[O_IDX(m, k)] = (float16_t)(O_fp32[k] / row_sumexp);
  }
}
