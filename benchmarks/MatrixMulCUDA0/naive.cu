// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA0/naive.cuh"

template <typename T>
__global__ void gemm_kernel(T *A, T *B, T *C, int m, int n, int k) {
  // Compute thread ID and corresponding matrix element
  long int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > m * n) return;

  int x = tid % m;
  int y = tid / m;

  if (x < m && y < n) {
    // Compute dot product of row of A and column of B
    float value = 0.f;
    for (int i = 0; i < k; i++) {
      value = value + (float)(A[x * k + i] * B[i * n + y]);
    }
    // Update matrix C
    C[x * n + y] = (T)value;
  }
}

template <size_t threadsPerBlock, typename T>
void GEMM0(T *dA, T *dB, T *dC, int m, int n, int k) {
  int blocks = ceil((float)m * n / threadsPerBlock);

  gemm_kernel<<<blocks, threadsPerBlock>>>(dA, dB, dC, m, n, k);
  cudaDeviceSynchronize();
}

template void GEMM0<TPB, float>(float *dA, float *dB, float *dC, int m, int n,
                                int k);
template void GEMM0<TPB, half>(half *dA, half *dB, half *dC, int m, int n,
                               int k);
