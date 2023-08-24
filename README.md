## introduction
This repository showcases various features of GEMM aimed at enhancing its performance. 
```
C = alpha * A * B + beta * C
```
## Matrix Multiplication Algorithm Implementations

* MatrixMulCUDA0
    * Naive GEMM implementation.
* MatrixMulCUDA1
    * Utilizing warp/block for fused multiply-add (FMA) calculations.
* MatrixMulCUDA2
    * Loading data with strides from global memory to shared memory.
* MatrixMulCUDA3
    * Aligning shared memory for optimized memory access.
* MatrixMulCUDA4
    * Loading data twice per thread for improved data reuse.
* MatrixMulCUDA5
    * Minimizing bank conflicts in shared memory accesses.
* MatrixMulCUDA6
    * Using ping-pong buffer strategy.
* MatrixMulCUDA7
    * Implementing fast 128x128 block GEMM. (Note: A bug causing segment faults needs to be fixed.)
* MatrixMulCUDA8
    * desen refer to https://github.com/Cjkkkk/CUDA_gemm/blob/master/src/cuda/dense.cu
* MatrixMulCUDA9
    * Implementation using cuBLAS.
* MatrixMulCUDA10
    * yzaiust refer to  https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
* MatrixMulCUDA11
    * yinghan resfer to https://github.com/Yinghan-Li/YHs_Sample/blob/master/cuda/gemm/sgemm.cu


## TODO
* Address the bug causing a segment fault in MatrixMulCUDA7.
* Refactor and introduce a baseGemm class.
* Correct the TFlops miscalculation.
* Fix the issue where CUDA implementations 0 to 6 cannot handle cases where m ≠ n ≠ k.
