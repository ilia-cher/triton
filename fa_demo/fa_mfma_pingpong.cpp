#include "fa_kernels.h"

#include <hip/hip_runtime.h>

__launch_bounds__(FA_MFMA_NTHREADS, 1)
__global__ void fa_mfma_pingpong(
    const float16_t *Qb,
    const float16_t *Kb,
    const float16_t *Vb,
    float16_t *Ob) {

  const float16_t *Q = Qb + blockIdx.y * NCTX * DHEAD;
  const float16_t *K = Kb + blockIdx.y * NCTX * DHEAD;
  const float16_t *V = Vb + blockIdx.y * NCTX * DHEAD;
  float16_t *O = Ob + blockIdx.y * NCTX * DHEAD;

  bool isFirst = threadIdx.y / 4 == 0;

  const int m = blockIdx.x * FA_MFMA_BLOCK_M + threadIdx.y * 32;
  const float16_t *Qstripe = Q + Q_IDX(m, 0);
  const float16_t *Kblock = K;
  const float16_t *Vblock = V;

  __shared__ float16_t Ks[32 * 8 * FA_MFMA_K_BLOCKS];
  __shared__ float16_t Vs[8 * 32 * FA_MFMA_K_BLOCKS];

  float16x4 Qreg[FA_MFMA_K_BLOCKS];

  float M = -std::numeric_limits<float>::infinity();
  float L = {0.0};

  floatx16 accO[FA_MFMA_O_NACC] = {{0.0}};

  ////

  if (isFirst) {

    for (int nBlockIdx = 0; nBlockIdx < CEIL_DIV(NCTX, 32); ++nBlockIdx) {
      const float16_t *Qtile = Qstripe;
      const float16_t *Ktile = Kblock + K_IDX(32 * nBlockIdx, 0);
      float16_t *Kstile = Ks;

      // Preload K
      for (int kOuterBlockIdx = 0; 
              kOuterBlockIdx < FA_MFMA_K_BLOCKS;
              kOuterBlockIdx += 4) {      
        const int ksx = 2 * (threadIdx.y % 4) + threadIdx.x / 32;
        const int ksy = threadIdx.x % 32;
        // Only half of the waves preload K, each wave loads 2 lines at once:
        const int stepLines = FA_MFMA_NWAVES;
        for (int lines = 0; lines < 32; lines += stepLines) {
          Kstile[R32_IDX(lines + ksx, ksy)] = Ktile[K_IDX(lines + ksx, ksy)];  
        }

        Ktile += K_IDX(0, 32);
        Kstile += R32_IDX(32, 0);
      }
      __syncthreads();

      Kstile = Ks;

      // First GEMM
      floatx16 acc = {0.0};
      for (int kBlockIdx = 0;
              kBlockIdx < FA_MFMA_K_BLOCKS;
              ++kBlockIdx) {

        if (nBlockIdx == 0) {
          mfma_reg_load_Qtile(Qreg[kBlockIdx], Qtile);
        }

        float16x4 Kreg;
        mfma_reg_load_Kstile(Kreg, Kstile);

        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(Kreg, Qreg[kBlockIdx], acc, 0, 0, 0);
      
        Qtile += Q_IDX(0, 8);
        Kstile += R32_IDX(0, 8);
        if ((kBlockIdx + 1) % 4 == 0) {
          Kstile += R32_IDX(31, 0);
        }
      }

      float16x4 accP[4];

      // Online softmax
      float rowmax1 = -std::numeric_limits<float>::infinity();
      for (int idx = 0; idx < 16; ++idx) {
        acc[idx] *= ISQRTD;
        if (rowmax1 < acc[idx]) {
          rowmax1 = acc[idx];
        }
      }
      float rowmax2 = __shfl(rowmax1, (threadIdx.x + 32) % 64);
      float rowmax = (rowmax1 > rowmax2) ? rowmax1 : rowmax2;
      float Mnew = (rowmax > M) ? rowmax : M;
      float alpha = std::expf(M - Mnew);

      M = Mnew;
      L *= alpha;

      float rowsum1 = 0;
      for (int idx = 0; idx < 16; ++idx) {
        acc[idx] = std::expf(acc[idx] - M);
        rowsum1 += acc[idx];
        accP[idx / 4][idx % 4] = (float16_t)acc[idx];
      }
      float rowsum2 = __shfl(rowsum1, (threadIdx.x + 32) % 64);
      float rowsum = rowsum1 + rowsum2;
      L += rowsum;
      
      // Don't preload V, wait for the other half to finish preloading
      __syncthreads();

      float16_t *Vstile = Vs;

      // Second GEMM
      for (int outerIdx = 0; outerIdx < FA_MFMA_O_NACC; ++outerIdx) {
        // Correct accO using the new running max
        for (int accIdx = 0; accIdx < 16; ++accIdx) {
          accO[outerIdx][accIdx] *= alpha;
        }

        for (int pIdx = 0; pIdx < 4; ++pIdx) {
          float16x4 Vreg;
          mfma_reg_load_Vstile(Vreg, Vstile);

          accO[outerIdx] = __builtin_amdgcn_mfma_f32_32x32x8f16(Vreg, accP[pIdx], accO[outerIdx], 0, 0, 0);

          Vstile += R32_IDX(8, 0);
        }
      }
    }  // nBlockIdx

  } else {  // isFirst

    for (int nBlockIdx = 0; nBlockIdx < CEIL_DIV(NCTX, 32); ++nBlockIdx) {
      const float16_t *Qtile = Qstripe;
      float16_t *Kstile = Ks;

      // Don't preload K, wait for the other half to finish preloading
      __syncthreads();

      // Preload V
      //
      const float16_t *Vtile = Vblock + V_IDX(32 * nBlockIdx, 0);
      float16_t *Vstile = Vs;
      for (int outerIdx = 0; outerIdx < FA_MFMA_O_NACC; ++outerIdx) {
        const int vsx = 2 * (threadIdx.y % 4) + threadIdx.x / 32;
        const int vsy = threadIdx.x % 32;
        const int stepLines = FA_MFMA_NWAVES;  // only half of the waves
        for (int lines = 0; lines < 32; lines += stepLines) {
          Vstile[R32_IDX(lines + vsx, vsy)] = Vtile[V_IDX(lines + vsx, vsy)];  
        }
        Vtile += V_IDX(0, 32);
        Vstile += R32_IDX(32, 0);
      }
      Vstile = Vs;

      Kstile = Ks;

      // First GEMM
      floatx16 acc = {0.0};
      for (int kBlockIdx = 0;
              kBlockIdx < FA_MFMA_K_BLOCKS;
              ++kBlockIdx) {

        if (nBlockIdx == 0) {
          mfma_reg_load_Qtile(Qreg[kBlockIdx], Qtile);
        }

        float16x4 Kreg;
        mfma_reg_load_Kstile(Kreg, Kstile);

        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(Kreg, Qreg[kBlockIdx], acc, 0, 0, 0);
      
        Qtile += Q_IDX(0, 8);
        Kstile += R32_IDX(0, 8);
        if ((kBlockIdx + 1) % 4 == 0) {
          Kstile += R32_IDX(31, 0);
        }
      }

      __syncthreads();

      float16x4 accP[4];

      // Online softmax
      float rowmax1 = -std::numeric_limits<float>::infinity();
      for (int idx = 0; idx < 16; ++idx) {
        acc[idx] *= ISQRTD;
        if (rowmax1 < acc[idx]) {
          rowmax1 = acc[idx];
        }
      }
      float rowmax2 = __shfl(rowmax1, (threadIdx.x + 32) % 64);
      float rowmax = (rowmax1 > rowmax2) ? rowmax1 : rowmax2;
      float Mnew = (rowmax > M) ? rowmax : M;
      float alpha = std::expf(M - Mnew);

      M = Mnew;
      L *= alpha;

      float rowsum1 = 0;
      for (int idx = 0; idx < 16; ++idx) {
        acc[idx] = std::expf(acc[idx] - M);
        rowsum1 += acc[idx];
        accP[idx / 4][idx % 4] = (float16_t)acc[idx];
      }
      float rowsum2 = __shfl(rowsum1, (threadIdx.x + 32) % 64);
      float rowsum = rowsum1 + rowsum2;
      L += rowsum;
      
      // Second GEMM
      for (int outerIdx = 0; outerIdx < FA_MFMA_O_NACC; ++outerIdx) {
        // Correct accO using the new running max
        for (int accIdx = 0; accIdx < 16; ++accIdx) {
          accO[outerIdx][accIdx] *= alpha;
        }

        for (int pIdx = 0; pIdx < 4; ++pIdx) {
          float16x4 Vreg;
          mfma_reg_load_Vstile(Vreg, Vstile);

          accO[outerIdx] = __builtin_amdgcn_mfma_f32_32x32x8f16(Vreg, accP[pIdx], accO[outerIdx], 0, 0, 0);

          Vstile += R32_IDX(8, 0);
        }
      }
    }  // nBlockIdx
  }

  float16_t *Otile = O + O_IDX(m, 0);
  // Normalize and output tiles of accO
  // (transposed MFMA layout -> blocked non-transposed)
  for (int outerIdx = 0; outerIdx < FA_MFMA_O_NACC; ++outerIdx) {
    const int osx = threadIdx.x % 32;
    for (int accIdx = 0; accIdx < 16; ++accIdx) {
      const int osy = (4 * (threadIdx.x / 32)) + (8 * (accIdx / 4)) + (accIdx % 4);
      Otile[O_IDX(osx, osy)] = (float16_t)(accO[outerIdx][accIdx] / L);
    }
    Otile += O_IDX(0, 32);
  }
}
