#include "MatrixMulCUDA5/bankconflictavoider.cuh"

// a = mxk, b = kxn
template <typename T>
__global__ void gemm_kernel5(int m, int n, int k, T *a, T *b, T *c) {
  const int tx = (threadIdx.x % 16) * 2;
  const int ty = threadIdx.x / 16 * 2;
  const int bx = blockIdx.x * 64;
  const int by = blockIdx.y * 64;

  T *begin_a = a + by * k;
  T *begin_b = b + bx;
  T *end_a = begin_a + k;

  __shared__ T ashare[64][64];
  __shared__ T bshare[64][64];
  float sum0[2][2] = {0};
  float sum1[2][2] = {0};
  float sum2[2][2] = {0};
  float sum3[2][2] = {0};

  // bigger split
  for (T *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += 64, b_ptr += 64 * n) {
// Unroll load.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        ashare[ty + i][tx + j] = a_ptr[(ty + i) * k + tx + j];
        ashare[ty + i][tx + j + 32] = a_ptr[(ty + i) * k + tx + j + 32];
        ashare[ty + i + 32][tx + j] = a_ptr[(ty + 32 + i) * k + tx + j];
        ashare[ty + i + 32][tx + j + 32] =
            a_ptr[(ty + 32 + i) * k + tx + j + 32];

        bshare[ty + i][tx + j] = b_ptr[(ty + i) * n + tx + j];
        bshare[ty + i][tx + j + 32] = b_ptr[(ty + i) * n + tx + j + 32];
        bshare[ty + i + 32][tx + j] = b_ptr[(ty + i + 32) * n + tx + j];
        bshare[ty + i + 32][tx + j + 32] =
            b_ptr[(ty + i + 32) * n + tx + j + 32];
      }
    }
    __syncthreads();

// Unroll calc.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int subk = 0; subk < 64; ++subk) {
          sum0[i][j] += (float)(ashare[ty + i][subk] * bshare[subk][tx + j]);
          sum1[i][j] +=
              (float)(ashare[ty + i][subk] * bshare[subk][tx + j + 32]);
          sum2[i][j] +=
              (float)(ashare[ty + i + 32][subk] * bshare[subk][tx + j]);
          sum3[i][j] +=
              (float)(ashare[ty + i + 32][subk] * bshare[subk][tx + j + 32]);
        }
      }
    }
    __syncthreads();
  }

// Unroll set.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      c[(by + ty + i) * n + bx + tx + j] = (T)sum0[i][j];
      c[(by + ty + i) * n + bx + tx + 32 + j] = (T)sum1[i][j];
      c[(by + ty + i + 32) * n + bx + tx + j] = (T)sum2[i][j];
      c[(by + ty + i + 32) * n + bx + tx + 32 + j] = (T)sum3[i][j];
    }
  }
}
template <typename T>
void GEMM5(T *dA, T *dB, T *dC, int m, int n, int k) {
  /*  each thread load 16 ashare and 16 bshare calc 16 FMA
   *  avoiding shared mem bank conflict by stride 32
   *
   *  ashared
   *  t0 - 32*mem - t0 - 32*mem - t0
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  1. load 4 * 4 float each thread
   *  2. load Gmem -> Smem each thread
   *  3. clac 16 FMA each thread
   */

  dim3 block(256);
  dim3 grid(m / 64, n / 64);
  gemm_kernel5<T><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM5<float>(float *dA, float *dB, float *dC, int m, int n,
                           int k);
template void GEMM5<__half>(__half *dA, __half *dB, __half *dC, int m, int n,
                            int k);
