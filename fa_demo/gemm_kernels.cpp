#include "gemm_kernels.h"

#include <hip/hip_runtime.h>

void gemm_ref(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C) {
  #pragma omp parallel for
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float tmp = 0.0;
      for (int k = 0; k < K; ++k) {
        tmp += A[A_IDX(m, k)] * B[B_IDX(k, n)];
      }
      C[C_IDX(m, n)] = alpha * tmp + beta * C[C_IDX(m, n)];
    }
  }
}

__global__ void gemm_naive(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M && n < N) {
    float tmp = 0.0;
    for (int k = 0; k < K; ++k) {
      tmp += A[A_IDX(m, k)] * B[B_IDX(k, n)];
    }
    C[C_IDX(m, n)] = alpha * tmp + beta * C[C_IDX(m, n)];
  }
}

// As: [SMEM_BLOCK_M, SMEM_BLOCK_K]
#define SMEM_AS_IDX(I, J) ((I) * SMEM_BLOCK_K + (J))
// Bs: [SMEM_BLOCK_K, SMEM_BLOCK_N]
#define SMEM_BS_IDX(I, J) ((I) + (J) * SMEM_BLOCK_K)

__launch_bounds__(SMEM_NTHREADS, 1)
__global__ void gemm_smem(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C) {

  __shared__ float16_t As[SMEM_BLOCK_M * SMEM_BLOCK_K];
  __shared__ float16_t Bs[SMEM_BLOCK_K * SMEM_BLOCK_N];

  const float16_t *Astart = A + A_IDX(blockIdx.x * SMEM_BLOCK_M, 0);
  const float16_t *Bstart = B + B_IDX(0, blockIdx.y * SMEM_BLOCK_N);
  const float16_t *Bsw = Bs + SMEM_BS_IDX(0, threadIdx.y * SMEM_WAVE_NOUTPUTS);

  float results[SMEM_WAVE_NOUTPUTS] = {0.0};

  for (int kBlockStart = 0; kBlockStart < K; kBlockStart += SMEM_BLOCK_K) {
    As[SMEM_AS_IDX(threadIdx.x, threadIdx.y)] = Astart[A_IDX(threadIdx.x, threadIdx.y)];
    Bs[SMEM_BS_IDX(threadIdx.y, threadIdx.x)] = Bstart[B_IDX(threadIdx.y, threadIdx.x)];

    __syncthreads();

    Astart += A_IDX(0, SMEM_BLOCK_K);
    Bstart += B_IDX(SMEM_BLOCK_K, 0);

    for (int k = 0; k < SMEM_BLOCK_K; ++k) {
      float Atmp = As[SMEM_AS_IDX(threadIdx.x, k)];
      for (int outIdx = 0; outIdx < SMEM_WAVE_NOUTPUTS; ++outIdx) {
        results[outIdx] += Atmp * Bsw[SMEM_BS_IDX(k, outIdx)];
      }
    }

    __syncthreads();
  }

  const int m = blockIdx.x * SMEM_BLOCK_M + threadIdx.x;  // threadIdx.x == lane
  const int n = blockIdx.y * SMEM_BLOCK_N + threadIdx.y * SMEM_WAVE_NOUTPUTS;  //

  for (int outIdx = 0; outIdx < SMEM_WAVE_NOUTPUTS; ++outIdx) {
    C[C_IDX(m, n + outIdx)] = alpha * results[outIdx] + beta * C[C_IDX(m, n + outIdx)];
  }
}

#define MFMA_K_OUTER_SIZE 128
#define MFMA_K_BLOCKS ((MFMA_K_OUTER_SIZE) / 8)
#define MFMA_K_BLOCK_STEP 4

#if MFMA_K_BLOCKS % MFMA_K_BLOCK_STEP != 0
#error "Expected number of K blocks multiple of K block step in MFMA GEMM"
#endif

#define mfma_reg_load_Atile(reg, tile)      \
  {                                         \
    int tx = threadIdx.x % 32;              \
    int ty = 4 * (threadIdx.x / 32);        \
    reg[0] = tile[A_IDX(tx, ty)];           \
    reg[1] = tile[A_IDX(tx, ty + 1)];       \
    reg[2] = tile[A_IDX(tx, ty + 2)];       \
    reg[3] = tile[A_IDX(tx, ty + 3)];       \
  }

#define mfma_reg_load_Bstile(reg, tile)     \
  {                                         \
    int tx = threadIdx.x % 32;              \
    int ty = 4 * (threadIdx.x / 32);        \
    reg[0] = tile[R32_IDX(ty, tx)];         \
    reg[1] = tile[R32_IDX(ty + 1, tx)];     \
    reg[2] = tile[R32_IDX(ty + 2, tx)];     \
    reg[3] = tile[R32_IDX(ty + 3, tx)];     \
  }

#define mfma_acc_store_Ctile(acc, tile, first, alpha, beta)                  \
  {                                                                          \
    int cy = threadIdx.x % 32;                                               \
    for (int accIdx = 0; accIdx < 16; ++accIdx) {                            \
      int cx = (4 * (threadIdx.x / 32)) + (8 * (accIdx / 4)) + (accIdx % 4); \
      if (first) {                                                           \
        tile[C_IDX(cx, cy)] = beta * Ctile[C_IDX(cx, cy)];                   \
      }                                                                      \
      tile[C_IDX(cx, cy)] += alpha * acc[accIdx];                            \
    }                                                                        \
  }

__launch_bounds__(MFMA_NTHREADS, 1)
__global__ void gemm_mfma(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C) {

  const float16_t *Astripe = A + A_IDX(blockIdx.x * MFMA_BLOCK_M + threadIdx.y * 32, 0);
  const float16_t *Bblock = B + B_IDX(0, blockIdx.y * MFMA_BLOCK_N);
  float *Cstripe = C + C_IDX(blockIdx.x * MFMA_BLOCK_M + threadIdx.y * 32, blockIdx.y * MFMA_BLOCK_N);

  __shared__ float16_t Bs[8 * 32 * MFMA_K_BLOCK_STEP];

  //for (int kOuter = 0; kOuter < K; kOuter += MFMA_K_OUTER_SIZE) {
  {
    int kOuter = 0;

    float16x4 Areg[MFMA_K_BLOCKS];
    for (int nBlockIdx = 0; nBlockIdx < CEIL_DIV(MFMA_BLOCK_N, 32); ++nBlockIdx) {
      const float16_t *Atile = Astripe + A_IDX(0, kOuter);
      const float16_t *Btile = Bblock + B_IDX(kOuter, 32 * nBlockIdx);
      float *Ctile = Cstripe + C_IDX(0, 32 * nBlockIdx);

      floatx16 acc = {0.0};
      for (int kOuterBlockIdx = 0; 
               kOuterBlockIdx < MFMA_K_BLOCKS;
               kOuterBlockIdx += MFMA_K_BLOCK_STEP) {

        // Waves cooperate to (pre)load MFMA_K_BLOCK_STEP tiles of B into
        // shared memory, e.g.:
        // 4 waves x 64 threads == 256 loads == 32x8 elements (single tile)
        // (8 waves - 2 tiles)

        float16_t *Bstile_wg = Bs;
        const float16_t *Btile_wg = Btile;
        int bsx = 2 * threadIdx.y + threadIdx.x / 32;
        int bsy = threadIdx.x % 32;

        const int bsIters = MFMA_K_BLOCK_STEP / (MFMA_NWAVES / 4);
        const int bsStep = 2 * MFMA_NWAVES;

        for (int bsIdx = 0; bsIdx < bsIters; ++bsIdx) {
          Bstile_wg[R32_IDX(bsx, bsy)] = Btile_wg[B_IDX(bsx, bsy)];
          Bstile_wg += R32_IDX(bsStep, 0);
          Btile_wg += B_IDX(bsStep, 0);
        }
/*
        float16_t *Bstile_wg = Bs;
        const float16_t *Btile_wg = Btile;
        int bsx = threadIdx.x % 32;
        int bsy = 2 * threadIdx.y + threadIdx.x / 32;

        for (int it = 0; it < 2; ++it) {
          Bstile_wg[R32_IDX(bsx, bsy)] = Btile_wg[B_IDX(bsx, bsy)];
          Bstile_wg += R32_IDX(0, 16);
          Btile_wg += B_IDX(0, 16);
        }
*/
        Btile += B_IDX(8 * MFMA_K_BLOCK_STEP, 0);
        float16_t *Bstile = Bs;

        __syncthreads();

        for (int kBlockIdx = kOuterBlockIdx;
                 kBlockIdx < kOuterBlockIdx + MFMA_K_BLOCK_STEP;
                 ++kBlockIdx) {

          if (nBlockIdx == 0) {
            mfma_reg_load_Atile(Areg[kBlockIdx], Atile);
          }

          float16x4 Breg;
          mfma_reg_load_Bstile(Breg, Bstile);

          acc = __builtin_amdgcn_mfma_f32_32x32x8f16(Areg[kBlockIdx], Breg, acc, 0, 0, 0);
        
          Atile += A_IDX(0, 8);
          Bstile += R32_IDX(8, 0);
        }

        __syncthreads();
      }

      mfma_acc_store_Ctile(acc, Ctile, kOuter == 0, alpha, beta);
    }
  }
}
