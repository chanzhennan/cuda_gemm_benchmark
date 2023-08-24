## introduction
A repository showcasing various features of GEMM to enhance its performance. 
```
C = alpha * A * B + beta * C
```
## algorithm

* MatrixMulCUDA0
    * naive gemm implememt
* MatrixMulCUDA1
    * using warp/block to calc FMA
* MatrixMulCUDA2
    * load stride Gmem to Smem
* MatrixMulCUDA3
    * aligning shared memory
* MatrixMulCUDA4
    * load twice each thread
* MatrixMulCUDA5
    * avoiding bankconflict
* MatrixMulCUDA6
    * using pingpang buffer
* MatrixMulCUDA7
    * implement fast 128*128 block gemm
* MatrixMulCUDA8
    * desen refer to https://github.com/Cjkkkk/CUDA_gemm/blob/master/src/cuda/dense.cu
* MatrixMulCUDA9
    * cublas implement
* MatrixMulCUDA10
    * yzaiust refer to  https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
* MatrixMulCUDA11
    * yinghan resfer to https://github.com/Yinghan-Li/YHs_Sample/blob/master/cuda/gemm/sgemm.cu


## TODO
* (MatrixMulCUDA7) fix bug,  segment fault occur
* refactor, add baseGemm class
* fix bug, TFlops miscalc
* fix bug, cuda0 - cuda 6 can not deal with m != n != k 
