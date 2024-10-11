#ifndef FA_KERNELS_H
#define FA_KERNELS_H

#include "fa_common.h"

void attn_ref(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

void fa_ref(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

#define FA_NAIVE_NWAVES 8
#define FA_NAIVE_NTHREADS (FA_NAIVE_NWAVES * 64)
#define FA_NAIVE_BLOCK_M (FA_NAIVE_NWAVES * 64)

//__launch_bounds__(FA_NAIVE_NTHREADS, 1)
__global__ void fa_naive(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

#define FA_MFMA_NWAVES 8
#define FA_MFMA_NTHREADS (FA_MFMA_NWAVES * 64)
#define FA_MFMA_BLOCK_M (FA_MFMA_NWAVES * 32)

#define FA_MFMA_K_BLOCKS (DHEAD / 8)
#define FA_MFMA_O_NACC (DHEAD / 32)

#define mfma_reg_load_Qtile(reg, tile)      \
  {                                         \
    int tx = threadIdx.x % 32;              \
    int ty = 4 * (threadIdx.x / 32);        \
    reg[0] = tile[Q_IDX(tx, ty)];           \
    reg[1] = tile[Q_IDX(tx, ty + 1)];       \
    reg[2] = tile[Q_IDX(tx, ty + 2)];       \
    reg[3] = tile[Q_IDX(tx, ty + 3)];       \
  }

#define mfma_reg_load_Kstile(reg, tile)        \
  {                                            \
    int tx = threadIdx.x % 32;                 \
    int ty = 4 * (threadIdx.x / 32);           \
    reg[0] = tile[R32_IDX(tx, ty)];            \
    reg[1] = tile[R32_IDX(tx, ty + 1)];        \
    reg[2] = tile[R32_IDX(tx, ty + 2)];        \
    reg[3] = tile[R32_IDX(tx, ty + 3)];        \
  }

#define mfma_reg_load_Vstile(reg, tile)        \
  {                                            \
    int tx = 4 * (threadIdx.x / 32);           \
    int ty = threadIdx.x % 32;                 \
    reg[0] = tile[R32_IDX(tx, ty)];            \
    reg[1] = tile[R32_IDX(tx + 1, ty)];        \
    reg[2] = tile[R32_IDX(tx + 2, ty)];        \
    reg[3] = tile[R32_IDX(tx + 3, ty)];        \
  }

//__launch_bounds__(FA_MFMA_NTHREADS, 1)
__global__ void fa_mfma(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

//__launch_bounds__(FA_MFMA_NTHREADS, 1)
__global__ void fa_mfma_pingpong(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

#endif  // FA_KERNELS_H
