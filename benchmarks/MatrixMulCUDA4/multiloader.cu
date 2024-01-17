#include "MatrixMulCUDA4/multiloader.cuh"

// a = mxk, b = kxn
template <int BLOCK, int STRIDE, typename T>
__global__ void gemm_kernel4(int m, int n, int k, T *a, T *b, T *c) {
  // blockIdx control subpanel matrix
  constexpr int STEP = BLOCK * STRIDE;
  const int tx = threadIdx.x * STRIDE;
  const int ty = threadIdx.y * STRIDE;
  const int bx = blockIdx.x * STEP;
  const int by = blockIdx.y * STEP;

  T *begin_a = a + by * k;
  T *begin_b = b + bx;
  T *end_a = begin_a + k;

  float sum[STRIDE][STRIDE] = {0.f};

  // bigger stash for more data load
  __shared__ T ashare[STEP][2 * STEP];
  __shared__ T bshare[2 * STEP][STEP];

  for (T *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += 2 * STEP, b_ptr += 2 * STEP * n) {
    for (int i = 0; i < STRIDE; ++i) {
      for (int j = 0; j < STRIDE; ++j) {
        // load twice each thread
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
          sum[i][j] += (float)(ashare[ty + i][kk] * bshare[kk][tx + j]);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < STRIDE; ++i) {
    for (int j = 0; j < STRIDE; ++j) {
      c[(by + ty + i) * n + bx + tx + j] = (T)sum[i][j];
    }
  }
}

template <size_t BLOCK, typename T>
void GEMM4(T *dA, T *dB, T *dC, int m, int n, int k) {
  /*  (BLOCK * BLOCK) threads calc ((BLOCK + STRIDE) * (BLOCK + STRIDE)) data
   *
   *  t0 t1 t0 t1 - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  - - - - - - - - - - - -
   *  1. load 2 * 2 float each thread
   *  2. load Gmem -> Smem each thread
   *  3. clac 4 FMA each thread
   *  this kerenl STRIDE = 2
   */

  constexpr int STRIDE = 2;  // every thread calc STRIDExSTRIDE result
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
  gemm_kernel4<BLOCK, STRIDE, T><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM4<BLOCKSIZE, float>(float *dA, float *dB, float *dC, int m,
                                      int n, int k);

template void GEMM4<BLOCKSIZE, __half>(__half *dA, __half *dB, __half *dC,
                                       int m, int n, int k);
