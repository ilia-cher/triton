#include "fa_kernels.h"
#include "util.h"

#define FA_ITERS 25

enum class TestFAKernel {
  NAIVE,
  MFMA,
  MFMA_PP
};

void launchFA(
    TestFAKernel krnl,
    const float16_t *hQ,
    const float16_t *hK,
    const float16_t *hV,
    float16_t *hO) {
  int tSize = sizeof(float16_t) * BH * NCTX * DHEAD;
  float16_t *dQ = nullptr;
  HIP_CHECK(hipMalloc(&dQ, tSize));
  float16_t *dK = nullptr;
  HIP_CHECK(hipMalloc(&dK, tSize));
  float16_t *dV = nullptr;
  HIP_CHECK(hipMalloc(&dV, tSize));
  float16_t *dO = nullptr;
  HIP_CHECK(hipMalloc(&dO, tSize));

  HIP_CHECK(hipMemcpy(dQ, hQ, tSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, hK, tSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, hV, tSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dO, hO, tSize, hipMemcpyHostToDevice));

  for (int it = 0; it < FA_ITERS; ++it) {

    if (krnl == TestFAKernel::NAIVE) {
      dim3 gridDim(CEIL_DIV(NCTX, FA_NAIVE_BLOCK_M), BH, 1);
      dim3 blockDim(64, FA_NAIVE_NWAVES, 1);
      fa_naive <<<gridDim, blockDim>>> (
        dQ, dK, dV,
        dO
      );
    } else if (krnl == TestFAKernel::MFMA) {
      dim3 gridDim(CEIL_DIV(NCTX, FA_MFMA_BLOCK_M), BH, 1);
      dim3 blockDim(64, FA_MFMA_NWAVES, 1);
      fa_mfma <<<gridDim, blockDim>>> (
        dQ, dK, dV,
        dO
      );
    } else if (krnl == TestFAKernel::MFMA_PP) {
      dim3 gridDim(CEIL_DIV(NCTX, FA_MFMA_BLOCK_M), BH, 1);
      dim3 blockDim(64, FA_MFMA_NWAVES, 1);
      fa_mfma_pp <<<gridDim, blockDim>>> (
        dQ, dK, dV,
        dO
      );
    } else {
      std::cerr << "Error: unsupported kernel\n";
    }

  }

  HIP_CHECK(hipMemcpy(hO, dO, tSize, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(dO));
  HIP_CHECK(hipFree(dV));
  HIP_CHECK(hipFree(dK));
  HIP_CHECK(hipFree(dQ));
}


int main() {
  //
  // Initialization
  //
  srand(SEED);
  HIP_CHECK(hipSetDevice(0));

  float16_t *hQ = new float16_t[BH * NCTX * DHEAD];
  float16_t *hK = new float16_t[BH * NCTX * DHEAD];
  float16_t *hV = new float16_t[BH * NCTX * DHEAD];
  float16_t *hO_attn_ref = new float16_t[BH * NCTX * DHEAD];
  float16_t *hO_ref = new float16_t[BH * NCTX * DHEAD];
  float16_t *hO_naive = new float16_t[BH * NCTX * DHEAD];
  float16_t *hO_mfma = new float16_t[BH * NCTX * DHEAD];
  float16_t *hO_mfma_pp = new float16_t[BH * NCTX * DHEAD];

  std::cout << "Initializing tensors... ";
  initBuffer<float16_t>(hQ, BH * NCTX * DHEAD, 4.0);
  memcpy(hK, hQ, BH * NCTX * DHEAD * sizeof(float16_t));
  initBuffer<float16_t>(hV, BH * NCTX * DHEAD, 4.0);
  std::cout << "done\n\n";

  //
  // FP16 FA
  //
  std::cout << "FP16 FA: Nctx, Dhead = " << NCTX << ", " << DHEAD << "; BH = " << BH << "\n";

  std::cout << "Computing ref... " << std::flush;
  attn_ref(
    hQ, hK, hV,
    hO_attn_ref,
    /*firstM=*/ 64);
  fa_ref(
    hQ, hK, hV,
    hO_ref,
    /*firstM=*/ 64);
  std::cout << "done\n\n";

  launchFA(
    TestFAKernel::NAIVE,
    hQ, hK, hV,
    hO_naive);
  
  launchFA(
    TestFAKernel::MFMA,
    hQ, hK, hV,
    hO_mfma);

  launchFA(
    TestFAKernel::MFMA_PP,
    hQ, hK, hV,
    hO_mfma_pp);

  std::cout << "attn_ref/ref :  " << (checkFAResult(hO_attn_ref, hO_ref, /*firstM=*/ 64) ? "OK" : "FAIL") << "\n";
  std::cout << "naive/ref    :  " << (checkFAResult(hO_naive, hO_ref, /*firstM=*/ 64) ? "OK" : "FAIL") << "\n";
  std::cout << "mfma/ref     :  " << (checkFAResult(hO_mfma, hO_ref, /*firstM=*/ 64) ? "OK" : "FAIL") << "\n";
  std::cout << "mfma_pp/ref  :  " << (checkFAResult(hO_mfma_pp, hO_ref, /*firstM=*/ 64) ? "OK" : "FAIL") << "\n\n";

  //
  // Cleanup
  //
  delete[] hO_mfma_pp;
  delete[] hO_mfma;
  delete[] hO_naive;
  delete[] hO_ref;
  delete[] hO_attn_ref;
  delete[] hV;
  delete[] hK;
  delete[] hQ;

  return 0;
}
