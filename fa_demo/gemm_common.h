#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include "common.h"

#define ALPHA 0.75
#define BETA 0.1

#define MSIZE (16*1024)
#define NSIZE (16*1024)
#define KSIZE 128

// BLAS-standard -- by default matices are col-maj:
// "N": col-maj
// "T": row-maj

#define USE_ATRANS
//#define USE_BTRANS
//#define USE_CTRANS

// A: MxK
#ifdef USE_ATRANS
// A[MxK] is row-major:
#define AS1 KSIZE
#define AS2 1
#define LDA KSIZE
#define ATRANS rocblas_operation_transpose
#define ASTR "T"
// can be also reinterpreted as col-major A^T [KxM]:
#define ATS1 1
#define ATS2 KSIZE
#define LDAT KSIZE
#define ATTRANS rocblas_operation_none
#else
// A[MxK] is col-major:
#define AS1 1
#define AS2 MSIZE
#define LDA MSIZE
#define ATRANS rocblas_operation_none
#define ASTR "N"
// can be also reinterpreted as row-major A^T [KxM]:
#define ATS1 MSIZE
#define ATS2 1
#define LDAT MSIZE
#define ATTRANS rocblas_operation_transpose
#endif
#define A_IDX(I, J) ((I) * AS1 + (J) * AS2)
#define AT_IDX(I, J) ((I) * ATS1 + (J) * ATS2)

// B: KxN
#ifdef USE_BTRANS
// B[KxN] is row-major:
#define BS1 NSIZE
#define BS2 1
#define LDB NSIZE
#define BTRANS rocblas_operation_transpose
#define BSTR "T"
// can be also reinterpreted as col-major B^T [NxK]:
#define BTS1 1
#define BTS2 NSIZE
#define LDBT NSIZE
#define BTTRANS rocblas_operation_none
#else
// B[KxN] is col-major:
#define BS1 1
#define BS2 KSIZE
#define LDB KSIZE
#define BTRANS rocblas_operation_none
#define BSTR "N"
// can be also reinterpreted as row-major B^T [NxK]:
#define BTS1 KSIZE
#define BTS2 1
#define LDBT KSIZE
#define BTTRANS rocblas_operation_transpose
#endif
#define B_IDX(I, J) ((I) * BS1 + (J) * BS2)
#define BT_IDX(I, J) ((I) * BST1 + (J) * BST2)

// C: MxN
#ifdef USE_CTRANS
// C[MxN] is row-major:
#define CS1 NSIZE
#define CS2 1
#define LDC NSIZE
#define CSTR "T"
// can be also reinterpreted as col-major C^T [NxM]:
#define CTS1 1
#define CTS2 NSIZE
#define LDCT NSIZE
#else
// C[MxN] is col-major:
#define CS1 1
#define CS2 MSIZE
#define LDC MSIZE
#define CSTR "N"
// can be also reinterpreted as row-major C^T [NxM]:
#define CTS1 MSIZE
#define CTS2 1
#define LDCT MSIZE
#endif
#define C_IDX(I, J) ((I) * CS1 + (J) * CS2)
#define CT_IDX(I, J) ((I) * CTS1 + (J) * CTS2)

bool checkGemmResult(
    const int M, const int N,
    const float *res, const float *ref,
    const float atol = DEFAULT_ATOL,
    const float rtol = DEFAULT_RTOL);

#endif  // GEMM_COMMON_H
