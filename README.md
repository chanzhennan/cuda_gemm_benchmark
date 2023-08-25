## Introduction
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

#
## Installation
* Edit build.sh file
   * cmake -DCUDA_ARCH=/your/cuda/arch -DCUDA_TOOLKIT_ROOT_DIR=/local/cuda/path
* bash build.sh
  
  
 ![image](https://github.com/chanzhennan/cuda_gemm_benchmark/assets/7290453/e879009a-475e-4f05-9e51-7771d3d5b765)
  |---------------------------------------------------------------------------------------------------------------|

#

## Performance
Run on RTX 4070 Ti | Theoretical Performance: FP32 (float) 40.09 TFLOPS
   * Reference: [GeForce RTX 4070 Ti Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-4070-ti.c3950)
   
| Benchmark                                       | Time     | CPU      | Iterations | UserCounters                 |
|-------------------------------------------------|----------|----------|------------|------------------------------|
| Naive<float>/Gemm_float/5120/4096/4096           | 1731 ms  | 1731 ms  | 1          | TFlops=0.099244/s, operation=171.799G |
| Blocker<float>/Gemm_float/5120/4096/4096         | 103 ms   | 103 ms   | 6          | TFlops=1.66191/s, operation=1030.79G |
| Strider<float>/Gemm_float/5120/4096/4096         | 19.9 ms  | 19.9 ms  | 30         | TFlops=8.62941/s, operation=5.15396T |
| Aligner<float>/Gemm_float/5120/4096/4096         | 17.3 ms  | 17.3 ms  | 33         | TFlops=9.93519/s, operation=5.66936T |
| MultiLoader<float>/Gemm_float/5120/4096/4096     | 19.8 ms  | 19.8 ms  | 31         | TFlops=8.67294/s, operation=5.32576T |
| BcAvoider<float>/Gemm_float/5120/4096/4096       | 24.2 ms  | 24.2 ms  | 26         | TFlops=7.10627/s, operation=4.46677T |
| PpBuffer<float>/Gemm_float/5120/4096/4096        | 20.9 ms  | 20.9 ms  | 28         | TFlops=8.2018/s, operation=4.81036T |
| Dense<float>/Gemm_float/5120/4096/4096           | 11.0 ms  | 11.0 ms  | 61         | TFlops=15.5654/s, operation=10.4797T |
| Cublas<float>/Gemm_float/5120/4096/4096          | 5.95 ms  | 5.95 ms  | 115        | TFlops=28.8656/s, operation=19.7568T |
| Yzaiustc<float>/Gemm_float/5120/4096/4096        | 7.23 ms  | 7.23 ms  | 93         | TFlops=23.765/s, operation=15.9773T |
| Yhs<float>/Gemm_float/5120/4096/4096             | 6.78 ms  | 6.78 ms  | 100        | TFlops=25.3418/s, operation=17.1799T |



## Todo
* Address the bug causing a segment fault in MatrixMulCUDA7.
* Fix the issue where CUDA implementations 0 to 6 cannot handle cases where m = 8 n = 4096 k = 4096.

