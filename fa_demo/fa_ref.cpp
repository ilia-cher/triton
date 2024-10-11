#include "fa_kernels.h"

#include <hip/hip_runtime.h>
//#include <iostream>

void attn_ref(
    const float16_t *Qb,  // [BH x NCTX x DHEAD]
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob) {
  float *P = new float[NCTX * NCTX];
  float *O_fp32 = new float[NCTX * DHEAD];

  for (int bh = 0; bh < BH; ++bh) {
    const float16_t *Q = Qb + bh * NCTX * DHEAD;
    const float16_t *K = Kb + bh * NCTX * DHEAD;
    const float16_t *V = Vb + bh * NCTX * DHEAD;
    float16_t *O = Ob + bh * NCTX * DHEAD;

    #pragma omp parallel for
    for (int m = 0; m < NCTX; ++m) {
      for (int k = 0; k < DHEAD; ++k) {
        O_fp32[O_IDX(m, k)] = 0.0;
      }

      float row_max = -std::numeric_limits<float>::infinity();
      //float row_min = std::numeric_limits<float>::infinity();
      for (int n = 0; n < NCTX; ++n) {
        float tmp = 0.0;
        for (int k = 0; k < DHEAD; ++k) {
          tmp += Q[Q_IDX(m, k)] * K[K_IDX(n, k)];
        }
        tmp *= ISQRTD;

        P[P_IDX(m, n)] = tmp;

        if (row_max < tmp) {
          row_max = tmp;
        }
        //if (row_min > tmp) {
        //  row_min = tmp;
        //}
      }

      float row_sumexp = 0.0;
      for (int n = 0; n < NCTX; ++n) {
        float tmp = std::expf(P[P_IDX(m, n)] - row_max);
        row_sumexp += tmp;
        P[P_IDX(m, n)] = tmp;
      }
      for (int n = 0; n < NCTX; ++n) {
        P[P_IDX(m, n)] /= row_sumexp;
        float16_t tmp_fp16 = (float16_t) (P[P_IDX(m, n)]);
        //
        for (int k = 0; k < DHEAD; ++k) {
          O_fp32[O_IDX(m, k)] += tmp_fp16 * V[V_IDX(n, k)];
        }
      }

  #ifdef DEBUG
      if (m == 0) {
        std::cerr << "\n";
        for (int n = 0; n < NCTX && n < 8; ++n) {
          std::cerr << "P[" << m << "," << n << "] = " << P[P_IDX(m, n)] << std::endl;
        }
      }
  #endif

      //
      for (int k = 0; k < DHEAD; ++k) {
        O[O_IDX(m, k)] = (float16_t)O_fp32[O_IDX(m, k)];
      }
    }
  }

  delete[] O_fp32;
  delete[] P;
}

void fa_ref(
    const float16_t *Qb,  // [BH x NCTX x DHEAD]
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob) {
  float *O_fp32 = new float[NCTX * DHEAD];

  for (int bh = 0; bh < BH; ++bh) {
    const float16_t *Q = Qb + bh * NCTX * DHEAD;
    const float16_t *K = Kb + bh * NCTX * DHEAD;
    const float16_t *V = Vb + bh * NCTX * DHEAD;
    float16_t *O = Ob + bh * NCTX * DHEAD;

    //std::cout << (bh + 1) << " " << std::flush;

    #pragma omp parallel for
    for (int m = 0; m < NCTX; ++m) {
      for (int k = 0; k < DHEAD; ++k) {
        O_fp32[O_IDX(m, k)] = 0.0;
      }

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
            float t = O_fp32[O_IDX(m, k)];
            O_fp32[O_IDX(m, k)] *= alpha;
          }
          O_fp32[O_IDX(m, k)] += tmp_fp16 * V[V_IDX(n, k)];
        }
      }

      //
      for (int k = 0; k < DHEAD; ++k) {
        O[O_IDX(m, k)] = (float16_t)(O_fp32[O_IDX(m, k)] / row_sumexp);
      }
    }
  }

  delete[] O_fp32;
}
