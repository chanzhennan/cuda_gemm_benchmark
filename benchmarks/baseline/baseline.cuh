#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define TPB 128

#include <iostream>

template <size_t threadsPerBlock, typename T>
void GEMM(T *dA, T *dB, T*dC, int m, int n, int k);
