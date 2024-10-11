#!/bin/bash

hipcc gemm_demo.cpp gemm_common.cpp gemm_kernels.cpp -o gemm_demo --save-temps -Rpass-analysis=kernel-resource-usage -lrocblas -fopenmp ;

runTracer.sh -o results.rpd ./gemm_demo ; python print_results.py ;

/bin/bash ./cleanup.sh

