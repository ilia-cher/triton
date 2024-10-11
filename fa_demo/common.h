#ifndef COMMON_H
#define COMMON_H

#include <hip/hip_fp16.h>
using float16_t = _Float16;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

#define SEED 57

// as in the Triton tutorials
#define DEFAULT_ATOL 0.01
#define DEFAULT_RTOL 0.0

#define CEIL_DIV(A, B) ((A + B - 1) / (B))

#define R32_IDX(I, J) ((I) * 32 + (J))

#endif  // COMMON_H
