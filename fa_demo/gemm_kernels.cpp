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

#define MFMA_K_BLOCKS 16

#ifdef USE_ATRANS
#define mfma_load_A_tile_to_reg(reg, tile)        \
  {                                               \
    int tx = threadIdx.x % 32;                    \
    int ty = 4 * (threadIdx.x / 32);              \
    reg = *((float16x4*)(tile + A_IDX(tx, ty)));  \
  }
#else
#define mfma_load_A_tile_to_reg(reg, tile)  \
  {                                         \
    int tx = threadIdx.x % 32;              \
    int ty = 4 * (threadIdx.x / 32);        \
    reg[0] = tile[A_IDX(tx, ty)];           \
    reg[1] = tile[A_IDX(tx, ty + 1)];       \
    reg[2] = tile[A_IDX(tx, ty + 2)];       \
    reg[3] = tile[A_IDX(tx, ty + 3)];       \
  }
#endif

#define _BS_IDX(T, I, J) ((T) * 256 + (I) * 8 + (J))

#define mfma_load_BS_tile_to_reg(reg, tile)                    \
  {                                                            \
    int tx = threadIdx.x % 32;                                 \
    int ty = 4 * (threadIdx.x / 32);                           \
    reg = *((float16x4*)(tile + _BS_IDX(0, tx, ty)));          \
  }

#ifndef USE_BTRANS
#define mfma_load_B_stripe_to_reg(reg, tile)                                 \
  {                                                                          \
    const float16_t *B_wave_ptr = tile + B_IDX(0, threadIdx.y * 4);          \
    int bx = (threadIdx.x % 16) * 8;                                         \
    int by = threadIdx.x / 16;                                               \
    reg = *((float16x8*)(B_wave_ptr + B_IDX(bx, by)));                       \
  }
#else
#define mfma_load_B_stripe_to_reg(reg, tile)                                 \
  {                                                                          \
    const float16_t *B_wave_ptr = tile + B_IDX(0, threadIdx.y * 4);          \
    int bx = (threadIdx.x % 16) * 8;                                         \
    int by = threadIdx.x / 16;                                               \
    reg[0] = B_wave_ptr[B_IDX(bx, by)];                                      \
    reg[1] = B_wave_ptr[B_IDX(bx + 1, by)];                                  \
    reg[2] = B_wave_ptr[B_IDX(bx + 2, by)];                                  \
    reg[3] = B_wave_ptr[B_IDX(bx + 3, by)];                                  \
    reg[4] = B_wave_ptr[B_IDX(bx + 4, by)];                                  \
    reg[5] = B_wave_ptr[B_IDX(bx + 5, by)];                                  \
    reg[6] = B_wave_ptr[B_IDX(bx + 6, by)];                                  \
    reg[7] = B_wave_ptr[B_IDX(bx + 7, by)];                                  \
  }
#endif

#define mfma_store_reg_to_BS_stripe(reg, tile)          \
  {                                                     \
    int bst = threadIdx.x % 16;                         \
    int bsx = threadIdx.y * 4 + (threadIdx.x / 16);     \
    *((float16x8*)(tile + _BS_IDX(bst, bsx, 0))) = reg; \
  }

/*
#define mfma_acc_store_Ctile(acc, tile, first, alpha, beta)                      \
  {                                                                              \
    int cy = threadIdx.x % 32;                                                   \
    for (int accIdx = 0; accIdx < 16; ++accIdx) {                                \
      int cx = (4 * (threadIdx.x / 32)) + (8 * (accIdx / 4)) + (accIdx % 4);     \
      if (first) {                                                               \
        tile[C_IDX(cx, cy)] = beta * tile[C_IDX(cx, cy)] + alpha * acc[accIdx];  \
      } else {                                                                   \
        tile[C_IDX(cx, cy)] += alpha * acc[accIdx];                              \
      }                                                                          \
    }                                                                            \
  }
*/

#define mfma_acc_store_C_tile(acc, tile, first, alpha, beta)                      \
  {                                                                               \
    int cy = threadIdx.x % 32;                                                    \
    if (first) {                                                                  \
      for (int accIdx = 0; accIdx < 16; ++accIdx) {                               \
        int cx = (4 * (threadIdx.x / 32)) + (8 * (accIdx / 4)) + (accIdx % 4);    \
        tile[C_IDX(cx, cy)] = beta * tile[C_IDX(cx, cy)] + alpha * acc[accIdx];   \
      }                                                                           \
    } else {                                                                      \
      for (int accIdx = 0; accIdx < 16; ++accIdx) {                               \
        int cx = (4 * (threadIdx.x / 32)) + (8 * (accIdx / 4)) + (accIdx % 4);    \
        tile[C_IDX(cx, cy)] += alpha * acc[accIdx];                               \
      }                                                                           \
    }                                                                             \
  }

#define mfma_swap_ptr(ptr1, ptr2)   \
  {                                 \
    auto *tmp = ptr1;               \
    ptr1 = ptr2;                    \
    ptr2 = tmp;                     \
  }

__launch_bounds__(MFMA_NTHREADS, 1)
__global__ void gemm_mfma(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C) {

  int m = blockIdx.x * MFMA_BLOCK_M + threadIdx.y * 32;
  int n = blockIdx.y * MFMA_BLOCK_N;
  const float16_t *A_block = A + A_IDX(m, 0);
  const float16_t *B_block = B + B_IDX(0, n);
  float *C_tile = C + C_IDX(m, n);

  __shared__ float16_t BS[8 * 32 * MFMA_K_BLOCKS];
  __shared__ float16_t BS_next[8 * 32 * MFMA_K_BLOCKS];

  for (int kOuter = 0; kOuter < K; kOuter += MFMA_K_BLOCKS * 8) {

    const float16_t *A_stripe = A_block + A_IDX(0, kOuter);
    const float16_t *B_stripe = B_block + B_IDX(kOuter, 0);

    // A (16 32x8 tiles): HBM -> reg (unique per wave in a workgroup)
    float16x4 A_tile_reg[MFMA_K_BLOCKS];
    const float16_t *A_tile = A_stripe;
    for (int k = 0; k < MFMA_K_BLOCKS; ++k) {
      mfma_load_A_tile_to_reg(A_tile_reg[k], A_tile); A_tile += A_IDX(0, 8);
    }

    // B (16 32x8 tiles): HBM -> reg (shared by all 8 waves in a workgroup)
    // 64 threads, float16x8 - 2 tiles per wave per 128bit load
    float16x8 B_stripe_reg;
    mfma_load_B_stripe_to_reg(B_stripe_reg, B_stripe);

    // B: reg -> LDS
    float16_t *BS_stripe = BS;
    float16_t *BS_stripe_next = BS_next;
    mfma_store_reg_to_BS_stripe(B_stripe_reg, BS_stripe);

    __syncthreads();

    const int nBlocks = CEIL_DIV(MFMA_BLOCK_N, 32);
    for (int nBlockIdx = 0; nBlockIdx < nBlocks; ++nBlockIdx) {
      // Prefetch the next stipe of B (HBM -> reg)
      if (nBlockIdx < nBlocks - 1) {
        const float16_t *B_stripe_next = B_stripe + B_IDX(0, 32);
        mfma_load_B_stripe_to_reg(B_stripe_reg, B_stripe_next);
      }

      // Load the tiles of B from the shared memory into the registers
      float16x4 B_tile_reg[16];
      const float16_t *BS_tile = BS_stripe;
      for (int k = 0; k < MFMA_K_BLOCKS; ++k) {
        mfma_load_BS_tile_to_reg(B_tile_reg[k], BS_tile); BS_tile += _BS_IDX(1, 0, 0);
      }

      // Compute
      floatx16 acc = {0.0};
      for (int k = 0; k < MFMA_K_BLOCKS; ++k) {
        acc = __builtin_amdgcn_mfma_f32_32x32x8f16(A_tile_reg[k], B_tile_reg[k], acc, 0, 0, 0);
      }

      // Write out the resulting C tile
      mfma_acc_store_C_tile(acc, C_tile, kOuter == 0, alpha, beta);
      // and the next B stripe into LDS
      if (nBlockIdx < nBlocks - 1) {
        mfma_store_reg_to_BS_stripe(B_stripe_reg, BS_stripe_next)
      }

      __syncthreads();

      // Move to the next stripe of B and the output tile C;
      // swap the buffers
      B_stripe += B_IDX(0, 32);
      C_tile += C_IDX(0, 32);
      mfma_swap_ptr(BS_stripe, BS_stripe_next);
    }
  }
}
