#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define TPB 8

#include <iostream>

template <size_t BLOCK, typename T>
void GEMM4(T *dA, T *dB, T *dC, int m, int n, int k);
