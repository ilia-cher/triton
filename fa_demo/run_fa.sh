#!/bin/bash

hipcc fa_demo.cpp fa_common.cpp fa_ref.cpp fa_naive.cpp fa_mfma.cpp fa_mfma_pp.cpp -o fa_demo --save-temps -Rpass-analysis=kernel-resource-usage -lrocblas -fopenmp ;

runTracer.sh -o results.rpd ./fa_demo ; python print_results.py -w 5 -f results.rpd;

/bin/bash ./cleanup.sh
