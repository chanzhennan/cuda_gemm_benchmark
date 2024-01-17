#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define BLOCKSIZE 8

#include <iostream>

template <size_t BLOCK, typename T>
void GEMM6(T *dA, T *dB, T *dC, int m, int n, int k);
