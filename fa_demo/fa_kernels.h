#ifndef FA_KERNELS_H
#define FA_KERNELS_H

#include "fa_common.h"

void attn_ref(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob,
    int firstM = 0);

void fa_ref(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob,
    int firstM = 0);

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

#define mfma_load_Q_tile_to_reg(reg, tile)        \
  {                                               \
    int tx = threadIdx.x % 32;                    \
    int ty = 4 * (threadIdx.x / 32);              \
    reg = *((float16x4*)(tile + Q_IDX(tx, ty)));  \
    reg[0] *= ISQRTD * LOG2_E;                    \
    reg[1] *= ISQRTD * LOG2_E;                    \
    reg[2] *= ISQRTD * LOG2_E;                    \
    reg[3] *= ISQRTD * LOG2_E;                    \
  }

#define mfma_load_K_stripe_to_reg(reg, tile)                         \
  {                                                                  \
    const float16_t *K_wave_ptr = tile + K_IDX(threadIdx.y * 4, 0);  \
    int kx = threadIdx.x / 16;                                       \
    int ky = (threadIdx.x % 16) * 8;                                 \
    reg = *((float16x8*)(K_wave_ptr + K_IDX(kx, ky)));               \
  }

#define _KS_IDX(T, I, J) ((T) * 256 + (I) * 8 + (J))

#define mfma_store_reg_to_KS_stripe(reg, tile)           \
  {                                                      \
    int kst = threadIdx.x % 16;                          \
    int ksx = threadIdx.y * 4 + (threadIdx.x / 16);      \
    *((float16x8*)(tile + _KS_IDX(kst, ksx, 0))) = reg;  \
  }

#define mfma_load_KS_tile_to_reg(reg, tile)            \
  {                                                    \
    int tx = threadIdx.x % 32;                         \
    int ty = 4 * (threadIdx.x / 32);                   \
    reg = *((float16x4*)(tile + _KS_IDX(0, tx, ty)));  \
  }

#define mfma_load_V_stripe_to_reg(reg, tile)                         \
  {                                                                  \
    const float16_t *V_wave_ptr = tile + V_IDX(0, threadIdx.y * 16); \
    int vx = (threadIdx.x % 4) * 8;                                  \
    int vy = threadIdx.x / 4;                                        \
    reg = *((float16x8*)(V_wave_ptr + V_IDX(vx, vy)));               \
  }

#define _VS_IDX(T1, T2, I, J) ((T1) * 1024 + (T2) * 256 + (I) * 8 + (J))

#define mfma_store_reg_to_VS_stripe(reg, tile)                  \
  {                                                             \
    int vst1 = threadIdx.y / 2;                                 \
    int vst2 = threadIdx.x % 4;                                 \
    int vsx = (threadIdx.y % 2) * 16 + (threadIdx.x / 4);       \
    *((float16x8*)(tile + _VS_IDX(vst1, vst2, vsx, 0))) = reg;  \
  }

#define mfma_load_VS_tile_to_reg(reg, tile)               \
  {                                                       \
    int vx = threadIdx.x % 32;                            \
    int vy = 4 * (threadIdx.x / 32);                      \
    reg = *((float16x4*)(tile + _VS_IDX(0, 0, vx, vy)));  \
  }

//__launch_bounds__(FA_MFMA_NTHREADS, 1)
__global__ void fa_mfma(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

//__launch_bounds__(FA_MFMA_NTHREADS, 1)
__global__ void fa_mfma_pp(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob);

#endif  // FA_KERNELS_H
