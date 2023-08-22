#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #define TPB 16
#define BLOCKSIZE 16

#include <iostream>

template <size_t BLOCK, typename T>
void GEMM2(T *dA, T *dB, T *dC, int m, int n, int k);
