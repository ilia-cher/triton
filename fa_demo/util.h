#ifndef UTIL_H
#define UTIL_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <vector>

#define CHECK_RET_CODE(call, ret_code)                                 \
  {                                                                    \
    if ((call) != ret_code) {                                          \
      std::cerr << "Failed in call: \"" << #call << "\"" << std::endl; \
      hipError_t hipError = hipGetLastError();                         \
      std::cerr << "Error(" << hipError << ", ";                       \
      std::cerr << hipGetErrorName(hipError) << "): ";                 \
      std::cerr << hipGetErrorString(hipError) << std::endl;           \
      std::abort();                                                    \
    }                                                                  \
  }
#define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(call)                              \
  {                                                            \
    rocblas_status status = (call);                            \
    if (status != rocblas_status_success) {                    \
      std::cerr << "rocBLAS error in call: \"" << #call        \
                << "\"" << std::endl;                          \
      std::cerr << "Error: ";                                  \
      if (status == rocblas_status_invalid_handle) {           \
        std::cerr << "rocblas_status_invalid_handle";          \
      } else if (status == rocblas_status_not_implemented) {   \
        std::cerr << "rocblas_status_not_implemented";         \
      } else if (status == rocblas_status_invalid_pointer) {   \
        std::cerr << "rocblas_status_invalid_pointer";         \
      } else if (status == rocblas_status_invalid_size) {      \
        std::cerr << "rocblas_status_invalid_size";            \
      } else if (status == rocblas_status_memory_error) {      \
        std::cerr << "rocblas_status_memory_error";            \
      } else if (status == rocblas_status_internal_error) {    \
        std::cerr << "rocblas_status_internal_error";          \
      } else {                                                 \
        std::cerr << "(unknown)";                              \
      }                                                        \
      std::cerr << std::endl;                                  \
      std::abort();                                            \
    }                                                          \
  }
#endif

template <typename T>
void initBuffer(T *buf, size_t N, float maxValue = 1.0) {
  for (size_t idx = 0; idx < N; ++idx) {
    buf[idx] = static_cast<T>(maxValue * rand() / ((float)RAND_MAX));
  }
}

template <typename T>
bool checkResultWithStrides(
    const int M, const int N,
    const T *res, const T *ref,
    const int S1, const int S2,
    const float atol,
    const float rtol,
    const int nout = 8) {
  std::atomic<int> err_cnt = 0;
  std::mutex m_cnt;
  bool first = true;

  std::vector<float> diff_sum;
  std::vector<float> diff_min;
  std::vector<float> diff_max;
  diff_sum.resize(M, 0.0);
  diff_min.resize(M, std::numeric_limits<float>::infinity());
  diff_max.resize(M, -std::numeric_limits<float>::infinity());

  #pragma omp parallel for
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int idx = m * S1 + n * S2;
      float diff = abs((float)res[idx] - ref[idx]);
      diff_sum[m] += diff;
      if (diff < diff_min[m]) {
        diff_min[m] = diff;
      }
      if (diff > diff_max[m]) {
        diff_max[m] = diff;
      }
      float maxDiff = atol + rtol * abs((float)ref[idx]);
      if (isnan((float)res[idx]) || isnan((float)ref[idx]) ||
          isinf((float)res[idx]) || isinf((float)ref[idx]) ||
          diff > maxDiff) {
        int local_cnt = err_cnt.fetch_add(1);
        if (local_cnt < nout) {
          std::lock_guard<std::mutex> guard(m_cnt);
          if (first) {
            std::cerr << "\n";
          }
          first = false;
          std::cerr << std::fixed << std::setprecision(16)
                    << "[" << m << "," << n << "]:\t "
                    << (float)res[idx] << " "
                    << (float)ref[idx] << " "
                    << diff << " "
                    << maxDiff
                    << "\n";
        }
      }
    }
  }

  if (err_cnt != 0) {
    std::cerr << "#failed = " << err_cnt << "\n";
  } else {
#ifdef DEBUG
    std::cerr << "\n";
    for (int n = 0; n < N && n < nout; ++n) {
      int idx = /* 0 * S1 + */ n * S2;
      float diff = abs((float)res[idx] - ref[idx]);
      float maxDiff = atol + rtol * abs((float)ref[idx]);
      std::cerr << std::fixed << std::setprecision(16)
                << "[0," << n << "]:\t "
                << (float)res[idx] << " "
                << (float)ref[idx] << " "
                << diff << " "
                << maxDiff
                << "\n";
    }
    float d_sum = 0.0;
    float d_min = std::numeric_limits<float>::infinity();
    float d_max = -std::numeric_limits<float>::infinity();
    for (int m = 0; m < M; ++m) {
      d_sum += diff_sum[m];
      if (diff_min[m] < d_min) {
        d_min = diff_min[m];
      }
      if (diff_max[m] > d_max) {
        d_max = diff_max[m];
      }
    }
    std::cerr << "min abs diff: " << d_min << "\n";
    std::cerr << "max abs diff: " << d_max << "\n";
    std::cerr << "avg abs diff: " << (d_sum / (M * N)) << "\n";
#endif
  }
  return err_cnt == 0;
}

#endif  // UTIL_H
