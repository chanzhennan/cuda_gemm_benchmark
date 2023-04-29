#include "outer_memory/outer_memory.cuh"

// a = mxk, b = kxn
template <int BLOCK, int STRIDE>
__global__ void gemm_kernel5(int m, int n, int k, float *a, float *b,
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

  __shared__ float ashare[STEP][2 * STEP];
  __shared__ float bshare[2 * STEP][STEP];
  // bigger split
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += 2 * STEP, b_ptr += 2 * STEP * n) {
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        ashare[ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];
        ashare[ty + i][tx + j + STEP] = a_ptr[(ty + i) * k + tx + j + STEP];

        bshare[ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];
        bshare[ty + i + STEP][tx + j] = b_ptr[(ty + i + STEP) * n + tx + j];
      }
    }
    __syncthreads();

    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        for (int kk = 0; kk < 2 * STEP; ++kk) {
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
void GEMM5(T *dA, T *dB, T *dC, int m, int n, int k) {
  constexpr int STRIDE = 2;  // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  gemm_kernel5<BLOCK, STRIDE><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM5<TPB, float>(float *dA, float *dB, float *dC, int m, int n,
                                int k);
// template void GEMM5<TPB, int>(int *dA, int *dB, int *dC, int m, int n, int
// k);
