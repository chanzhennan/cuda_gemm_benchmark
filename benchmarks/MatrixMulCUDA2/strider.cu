#include "MatrixMulCUDA2/strider.cuh"

// a = mxk, b = kxn
template <int BLOCK, int STRIDE, typename T>
__global__ void gemm_kernel2(int m, int n, int k, T *a, T *b, T *c) {
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
void GEMM2(T *dA, T *dB, T *dC, int m, int n, int k) {
  /*  (BLOCK * BLOCK) threads calc ((BLOCK + STRIDE) * (BLOCK + STRIDE)) data
   *
   *  t0 t1 t0 t1 - - - - -
   *  t2 t3 t2 t3 - - - - -
   *  - - - - - - - - - - -
   *  - - - - - - - - - - -
   *  - - - - - - - - - - -
   *  - - - - - - - - - - -
   *  1. clac (block+stride) * (block+stride) each warp
   *  2. load Gmem -> Smem each thread
   *  3. clac 2 FMA each thread
   *  this kerenl STRIDE = 2
   */

  constexpr int STRIDE = 2;  // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  gemm_kernel2<BLOCK, STRIDE><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM2<BLOCKSIZE, float>(float *dA, float *dB, float *dC, int m,
                                      int n, int k);
