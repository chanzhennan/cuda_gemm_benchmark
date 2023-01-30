#include "stride/stride.cuh"

// a = mxk, b = kxn
template <int BLOCK, int STRIDE, typename T>
__global__ void gemm_kernel3(int m, int n, int k, T *a, T *b, T *c) {
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  T *begin_a = a + by * STEP * k;
  T *begin_b = b + bx * STEP;
  T *end_a = begin_a + k;

  T sum[STRIDE][STRIDE] = {0.f};
  for (T *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += STEP, b_ptr += STEP * n) {
    __shared__ T ashare[STEP][STEP];
    __shared__ T bshare[STEP][STEP];

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[ty * STRIDE + i][tx * STRIDE + j] =
            a_ptr[(ty * STRIDE + i) * k + tx * STRIDE + j];
        bshare[ty * STRIDE + i][tx * STRIDE + j] =
            b_ptr[(ty * STRIDE + i) * n + tx * STRIDE + j];
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < STEP; ++kk) {
          sum[i][j] +=
              ashare[ty * STRIDE + i][kk] * bshare[kk][tx * STRIDE + j];
        }
      }
    }

    __syncthreads();
  }

  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(STEP * by + ty * STRIDE + i) * n + STEP * bx + tx * STRIDE + j] =
          sum[i][j];
    }
  }
}

template <size_t BLOCK, typename T>
void GEMM3(T *dA, T *dB, T *dC, int m, int n, int k) {
  constexpr int STRIDE = 2;  // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  gemm_kernel3<BLOCK, STRIDE><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM3<TPB, float>(float *dA, float *dB, float *dC, int m, int n,
                                int k);
// template void GEMM3<TPB, int>(int *dA, int *dB, int *dC, int m, int n, int
// k);
