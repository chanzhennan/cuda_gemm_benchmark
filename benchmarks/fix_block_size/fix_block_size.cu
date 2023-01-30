#include "fix_block_size/fix_block_size.cuh"

// a = mxk, b = kxn
template <int BLOCK, int STRIDE>
__global__ void gemm_kernel4(int m, int n, int k, float *a, float *b,
                             float *c) {
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x * STRIDE;
  const int ty = threadIdx.y * STRIDE;
  const int bx = blockIdx.x * STEP;
  const int by = blockIdx.y * STEP;

  float *begin_a = a + by * k;
  float *begin_b = b + bx;
  float *end_a = begin_a + k;

  float sum[STRIDE][STRIDE] = {0.f};
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += STEP, b_ptr += STEP * n) {
    __shared__ __align__(16 * 1024) float ashare[STEP][STEP];
    __shared__ __align__(16 * 1024) float bshare[STEP][STEP];

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];
        bshare[ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];
      }
    }
    __syncthreads();

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < STEP; ++kk) {
          sum[i][j] += ashare[ty + i][kk] * bshare[kk][tx + j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(by + ty + i) * n + bx + tx + j] = sum[i][j];
    }
  }
}

template <size_t BLOCK, typename T>
void GEMM4(T *dA, T *dB, T *dC, int m, int n, int k) {
  constexpr int STRIDE = 4;  // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  gemm_kernel4<BLOCK, STRIDE><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM4<TPB, float>(float *dA, float *dB, float *dC, int m, int n,
                                int k);
// template void GEMM4<TPB, int>(int *dA, int *dB, int *dC, int m, int n, int
// k);
