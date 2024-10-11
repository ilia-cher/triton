#include "gemm_kernels.h"
#include "util.h"

#include <rocblas/rocblas.h>

rocblas_handle handle;

enum class TestGemmKernel {
  ROCBLAS,
  NAIVE,
  SMEM,
  MFMA
};

void launchGemm(
    TestGemmKernel krnl,
    int M, int N, int K,
    float alpha,
    const float16_t *hA,
    const float16_t *hB,
    float beta,
    float *hC) {
  float16_t *dA = nullptr;
  HIP_CHECK(hipMalloc(&dA, sizeof(float16_t) * M * K));
  float16_t *dB = nullptr;
  HIP_CHECK(hipMalloc(&dB, sizeof(float16_t) * K * N));
  float *dC = nullptr;
  HIP_CHECK(hipMalloc(&dC, sizeof(float) * M * N));
  HIP_CHECK(hipMemcpy(dA, hA, sizeof(float16_t) * M * K, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB, sizeof(float16_t) * K * N, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, hC, sizeof(float) * M * N, hipMemcpyHostToDevice));

  if (krnl == TestGemmKernel::ROCBLAS) {
    // C[MxN] = A[MxK] * B[KxN]

#ifndef USE_CTRANS
    // C[MxN] is col-major
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(
      handle, ATRANS, BTRANS,
      M, N, K,
      &alpha,
      dA, rocblas_datatype_f16_r, LDA,
      dB, rocblas_datatype_f16_r, LDB,
      &beta,
      dC, rocblas_datatype_f32_r, LDC,
      dC, rocblas_datatype_f32_r, LDC,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0, 0
    ));
#else
    // C[MxN] needs to be row-major:
    // rocblas always outputs the result in col-major order,
    // to get the result in row-major order - compute
    //     C^T [NxM] = B^T [NxK] * A^T [KxM]
    // (instead of C = A*B),
    // the resulting C^T [NxM] is going to be in col-major order ==
    // same buffer can be reinterpreted as C [MxN] in row-major order
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(
      handle, BTTRANS, ATTRANS,
      N, M, K,
      &alpha,
      dB, rocblas_datatype_f16_r, LDBT,
      dA, rocblas_datatype_f16_r, LDAT,
      &beta,
      dC, rocblas_datatype_f32_r, LDCT,
      dC, rocblas_datatype_f32_r, LDCT,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0, 0
    ));
#endif
  } else if (krnl == TestGemmKernel::NAIVE) {
    dim3 gridDim(CEIL_DIV(M, NAIVE_BLOCK_M), CEIL_DIV(N, NAIVE_BLOCK_N), 1);
    dim3 blockDim(NAIVE_BLOCK_M, NAIVE_BLOCK_N, 1);
    gemm_naive <<<gridDim, blockDim>>> (
      M, N, K,
      alpha, dA, dB,
      beta, dC
    );
  } else if (krnl == TestGemmKernel::SMEM) {
    dim3 gridDim(CEIL_DIV(M, SMEM_BLOCK_M), CEIL_DIV(N, SMEM_BLOCK_N), 1);
    dim3 blockDim(64, SMEM_NWAVES, 1);
    gemm_smem <<<gridDim, blockDim>>> (
      M, N, K,
      alpha, dA, dB,
      beta, dC
    );
  } else if (krnl == TestGemmKernel::MFMA) {
    dim3 gridDim(CEIL_DIV(M, MFMA_BLOCK_M), CEIL_DIV(N, MFMA_BLOCK_N), 1);
    dim3 blockDim(64, MFMA_NWAVES, 1);
    gemm_mfma <<<gridDim, blockDim>>> (
      M, N, K,
      alpha, dA, dB,
      beta, dC
    );
  } else {
    std::cerr << "Error: unsupported kernel\n";
  }

  HIP_CHECK(hipMemcpy(hC, dC, sizeof(float) * M * N, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dA));
}

int main() {
  //
  // Initialization
  //
  srand(SEED);
  HIP_CHECK(hipSetDevice(0));
  rocblas_initialize();
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
  CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  constexpr int M = MSIZE;
  constexpr int N = NSIZE;
  constexpr int K = KSIZE;
  constexpr float alpha = ALPHA;
  constexpr float beta = BETA;

  float16_t *hA = new float16_t[M * K];
  float16_t *hB = new float16_t[K * N];
  float *hC_ref = new float[M * N];
  float *hC_naive = new float[M * N];
  float *hC_blas = new float[M * N];
  float *hC_smem = new float[M * N];
  float *hC_mfma = new float[M * N];

  std::cout << "Initializing tensors... ";
  initBuffer<float16_t>(hA, M * K);
  initBuffer<float16_t>(hB, K * N);
  initBuffer<float>(hC_ref, M * N);
  memcpy(hC_naive, hC_ref, M * N * sizeof(float));
  memcpy(hC_blas, hC_ref, M * N * sizeof(float));
  memcpy(hC_smem, hC_ref, M * N * sizeof(float));
  memcpy(hC_mfma, hC_ref, M * N * sizeof(float));
  std::cout << "done\n\n";

  //
  // FP16 GEMM
  //
  std::cout << "FP16 GEMM: M, N, K = " << M << ", " << N << ", " << K << ";  (AB: " << ASTR << BSTR << ", C: " << CSTR << ")\n";

  // Compute CPU reference
  std::cout << "Computing ref... " << std::flush;
  gemm_ref(
    M, N, K,
    alpha, hA, hB,
    beta, hC_ref
  );
  std::cout << "done\n\n";

  // Launch GPU kernels
  launchGemm(
    TestGemmKernel::NAIVE,
    M, N, K,
    alpha, hA, hB,
    beta, hC_naive
  );

  launchGemm(
    TestGemmKernel::ROCBLAS,
    M, N, K,
    alpha, hA, hB,
    beta, hC_blas
  );

  launchGemm(
    TestGemmKernel::SMEM,
    M, N, K,
    alpha, hA, hB,
    beta, hC_smem
  );

  launchGemm(
    TestGemmKernel::MFMA,
    M, N, K,
    alpha, hA, hB,
    beta, hC_mfma
  );

  std::cout << "rocBLAS/ref  :  " << (checkGemmResult(M, N, hC_blas, hC_ref) ? "OK" : "FAIL") << "\n";
  std::cout << "naive/ref    :  " << (checkGemmResult(M, N, hC_naive, hC_ref) ? "OK" : "FAIL") << "\n";
  std::cout << "smem/ref     :  " << (checkGemmResult(M, N, hC_smem, hC_ref) ? "OK" : "FAIL") << "\n";
  std::cout << "mfma/ref     :  " << (checkGemmResult(M, N, hC_mfma, hC_ref) ? "OK" : "FAIL") << "\n\n";

  //
  // Cleanup
  //
  delete[] hC_mfma;
  delete[] hC_smem;
  delete[] hC_blas;
  delete[] hC_naive;
  delete[] hC_ref;
  delete[] hB;
  delete[] hA;

  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

  return 0;
}
