#include "baseline/baseline.cuh"

template <typename T>
__global__ void gemm_kernel(T *A, T *B, T *C, int m, int n, int k) {
    // Compute thread ID and corresponding matrix element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        // Compute dot product of row of A and column of B
        T value = 0;
        for (int i = 0; i < k; i++) {
            value += A[row * k + i] * B[i * n + col];
        }
        // Update matrix C
        C[row * n + col] = value;
    }
}


template <size_t threadsPerBlock, typename T>
void GEMM(T *dA, T *dB, T*dC, int m, int n, int k) {
    int blocks = ceil((float)m * n / threadsPerBlock);
    gemm_kernel<<<blocks, threadsPerBlock>>>(dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();
}

template void GEMM<TPB, float>(float *dA, float *dB, float *dC, int m, int n, int k);
template void GEMM<TPB, int>(int *dA, int *dB, int *dC, int m, int n, int k);
