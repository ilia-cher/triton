#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include "gemm_common.h"

// CPU reference GEMM
void gemm_ref(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C);

// Naive GEMM
#define NAIVE_BLOCK_M 64
#define NAIVE_BLOCK_N 16
__global__ void gemm_naive(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C);

// Simple GEMM kernel w/ SMEM (re)use
#define SMEM_NWAVES 8
#define SMEM_NTHREADS (SMEM_NWAVES * 64)
#define SMEM_WAVE_NOUTPUTS 8

#define SMEM_BLOCK_M 64
#define SMEM_BLOCK_N (SMEM_NWAVES * SMEM_WAVE_NOUTPUTS)
#define SMEM_BLOCK_K SMEM_NWAVES

#if (SMEM_BLOCK_K != SMEM_NWAVES) || (SMEM_BLOCK_M != 64) || (SMEM_BLOCK_N != SMEM_BLOCK_M)
#error "Unsupported in this SMEM GEMM kernel"
#endif

//__launch_bounds__(SMEM_NTHREADS, 1)
__global__ void gemm_smem(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C);

// Simple GEMM kernel w/ MFMA, SMEM use,
// blocking and subiterations along M, N, K
#define MFMA_NWAVES 8
#define MFMA_NTHREADS (MFMA_NWAVES * 64)
#define MFMA_BLOCK_M (MFMA_NWAVES * 32)
#define MFMA_BLOCK_N 1024

#if MFMA_NWAVES != 4 && MFMA_NWAVES != 8
#error "Expected 4 or 8 waves in MFMA GEMM"
#endif

//__launch_bounds__(MFMA_NTHREADS, 1)
__global__ void gemm_mfma(
    int M, int N, int K,
    float alpha, const float16_t *A, const float16_t *B,
    float beta, float *C);

#endif  // GEMM_KERNELS_H
