#!/bin/bash

hipcc fa_demo.cpp fa_common.cpp fa_ref.cpp fa_naive.cpp fa_mfma.cpp fa_mfma_preload.cpp fa_mfma_pingpong.cpp -o fa_demo --save-temps -Rpass-analysis=kernel-resource-usage -lrocblas -fopenmp ;

runTracer.sh -o results.rpd ./fa_demo ; python print_results.py ;

hipcc gemm_demo.cpp gemm_common.cpp gemm_kernels.cpp -o gemm_demo --save-temps -Rpass-analysis=kernel-resource-usage -lrocblas -fopenmp ;

runTracer.sh -o results.rpd ./gemm_demo ; python print_results.py ;
