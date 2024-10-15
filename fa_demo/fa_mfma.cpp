#include "fa_kernels.h"

#include <hip/hip_runtime.h>

__launch_bounds__(FA_MFMA_NTHREADS, 1)
__global__ void fa_mfma(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob) {

  const float16_t *Q = Qb + blockIdx.y * NCTX * DHEAD;
  const float16_t *K = Kb + blockIdx.y * NCTX * DHEAD;
  const float16_t *V = Vb + blockIdx.y * NCTX * DHEAD;
  float16_t *O = Ob + blockIdx.y * NCTX * DHEAD;

  const int m = blockIdx.x * FA_MFMA_BLOCK_M + threadIdx.y * 32;
  const float16_t *Q_stripe = Q + Q_IDX(m, 0);
  const float16_t *K_stripe = K;
  const float16_t *V_stripe = V;

  __shared__ float16_t KS[32 * 8 * FA_MFMA_K_BLOCKS * 2];
  __shared__ float16_t KS_next[32 * 8 * FA_MFMA_K_BLOCKS * 2];
  __shared__ float16_t VS[8 * 32 * FA_MFMA_K_BLOCKS * 2];
  __shared__ float16_t VS_next[8 * 32 * FA_MFMA_K_BLOCKS * 2];

  float16x4 Q_tile_reg[FA_MFMA_K_BLOCKS];
  const float16_t *Q_tile = Q_stripe;
  for (int k = 0; k < FA_MFMA_K_BLOCKS; ++k) {
    mfma_load_Q_tile_to_reg(Q_tile_reg[k], Q_tile); Q_tile += Q_IDX(0, 8);
  }

  float M = -std::numeric_limits<float>::infinity();
  float L = {0.0};

  floatx16 accO[4] = {{0.0}};

  //
  float16x8 K_stripe_reg[2];
  mfma_load_K_stripe_to_reg(K_stripe_reg[0], K_stripe);
  mfma_load_K_stripe_to_reg(K_stripe_reg[1], K_stripe + K_IDX(32, 0));

  float16_t *KS_stripe = KS;
  float16_t *KS_stripe_next = KS_next;
  mfma_store_reg_to_KS_stripe(K_stripe_reg[0], KS_stripe);
  mfma_store_reg_to_KS_stripe(K_stripe_reg[1], KS_stripe + _KS_IDX(16, 0, 0));

  float16x8 V_stripe_reg[2];
  mfma_load_V_stripe_to_reg(V_stripe_reg[0], V_stripe);
  mfma_load_V_stripe_to_reg(V_stripe_reg[1], V_stripe + V_IDX(32, 0));

  float16_t *VS_stripe = VS;
  float16_t *VS_stripe_next = VS_next;
  mfma_store_reg_to_VS_stripe(V_stripe_reg[0], VS_stripe);
  mfma_store_reg_to_VS_stripe(V_stripe_reg[1], VS_stripe + _VS_IDX(4, 0, 0, 0));

  __syncthreads();
  //

  const int nBlocks = CEIL_DIV(NCTX, 64);
  for (int nBlockIdx = 0; nBlockIdx < nBlocks; ++nBlockIdx) {
    // Prefetch the next stripes of K and V from HBM into the registers
    if (nBlockIdx < nBlocks - 1) {
      const float16_t *K_stripe_next = K_stripe + K_IDX(64, 0);
      mfma_load_K_stripe_to_reg(K_stripe_reg[0], K_stripe_next);
      mfma_load_K_stripe_to_reg(K_stripe_reg[1], K_stripe_next + K_IDX(32, 0));

      const float16_t *V_stripe_next = V_stripe + V_IDX(64, 0);
      mfma_load_V_stripe_to_reg(V_stripe_reg[0], V_stripe_next);
      mfma_load_V_stripe_to_reg(V_stripe_reg[1], V_stripe_next + V_IDX(32, 0));
    }

    // First GEMM
    floatx16 acc[2] = {{0.0}};
    for (int nIter = 0; nIter < 2; ++nIter) {
      // Preload K from LDS into the registers
      float16x4 K_tile_reg[FA_MFMA_K_BLOCKS];
      const float16_t *KS_tile = KS_stripe + _KS_IDX(nIter * 16, 0, 0);
      for (int k = 0; k < FA_MFMA_K_BLOCKS; ++k) {
        mfma_load_KS_tile_to_reg(K_tile_reg[k], KS_tile); KS_tile += _KS_IDX(1, 0, 0);
      }

      for (int k = 0; k < FA_MFMA_K_BLOCKS; ++k) {
        acc[nIter] = __builtin_amdgcn_mfma_f32_32x32x8f16(K_tile_reg[k], Q_tile_reg[k], acc[nIter], 0, 0, 0);
      }
    }

    // Online softmax
    float alpha = 1.0;
    for (int nIter = 0; nIter < 2; ++nIter) {
      float rowmax1 = -std::numeric_limits<float>::infinity();
      for (int idx = 0; idx < 16; ++idx) {
        if (rowmax1 < acc[nIter][idx]) {
          rowmax1 = acc[nIter][idx];
        }
      }
      float rowmax2 = __shfl(rowmax1, (threadIdx.x + 32) % 64);
      float rowmax = (rowmax1 > rowmax2) ? rowmax1 : rowmax2;
      float Mnew = (rowmax > M) ? rowmax : M;
      alpha *= std::exp2f(M - Mnew);

      M = Mnew;
    }
    L *= alpha;

    float16x4 accP[2][4];
    for (int nIter = 0; nIter < 2; ++nIter) {
      float rowsum1 = 0;
      for (int idx = 0; idx < 16; ++idx) {
        float e_acc = std::exp2f(acc[nIter][idx] - M);
        rowsum1 += e_acc;
        accP[nIter][idx / 4][idx % 4] = (float16_t)e_acc;
      }
      float rowsum2 = __shfl(rowsum1, (threadIdx.x + 32) % 64);
      float rowsum = rowsum1 + rowsum2;
      L += rowsum;
    }

    // Write the next stripes of K and V from the registers into LDS
    if (nBlockIdx < nBlocks - 1) {
      mfma_store_reg_to_KS_stripe(K_stripe_reg[0], KS_stripe_next);
      mfma_store_reg_to_KS_stripe(K_stripe_reg[1], KS_stripe_next + _KS_IDX(16, 0, 0));

      mfma_store_reg_to_VS_stripe(V_stripe_reg[0], VS_stripe_next);
      mfma_store_reg_to_VS_stripe(V_stripe_reg[1], VS_stripe_next + _VS_IDX(4, 0, 0, 0));
    }

    // Second GEMM

    // Correct accO using the new running max
    for (int oIdx = 0; oIdx < 4; ++oIdx) {
      for (int accIdx = 0; accIdx < 16; ++accIdx) {
        accO[oIdx][accIdx] *= alpha;
      }
    }

    for (int nIter = 0; nIter < 2; ++nIter) {
      // Preload V from LDS into the registers
      float16x4 V_tile_reg[4][4];
      const float16_t *VS_tile = VS_stripe + _VS_IDX(nIter * 4, 0, 0, 0);
      for (int idx = 0; idx < 16; ++idx) {
        mfma_load_VS_tile_to_reg(V_tile_reg[idx / 4][idx % 4], VS_tile); VS_tile += _VS_IDX(0, 1, 0, 0);
      }

      for (int oIdx = 0; oIdx < 4; ++oIdx) {
        for (int pIdx = 0; pIdx < 4; ++pIdx) {
          accO[oIdx] = __builtin_amdgcn_mfma_f32_32x32x8f16(V_tile_reg[oIdx][pIdx], accP[nIter][pIdx], accO[oIdx], 0, 0, 0);
        }
      }
    }

    __syncthreads();

    K_stripe += K_IDX(64, 0);
    V_stripe += V_IDX(64, 0);
    swap_ptr(KS_stripe, KS_stripe_next);
    swap_ptr(VS_stripe, VS_stripe_next);
  }  // nBlockIdx

  float16_t *Otile = O + O_IDX(m, 0);
  // Normalize and output tiles of accO
  // (transposed MFMA layout -> blocked non-transposed)
  for (int oIdx = 0; oIdx < 4; ++oIdx) {
    const int osx = threadIdx.x % 32;
    for (int accIdx = 0; accIdx < 16; ++accIdx) {
      const int osy = (4 * (threadIdx.x / 32)) + (8 * (accIdx / 4)) + (accIdx % 4);
      Otile[O_IDX(osx, osy)] = (float16_t)(accO[oIdx][accIdx] / L);
    }
    Otile += O_IDX(0, 32);
  }
}
