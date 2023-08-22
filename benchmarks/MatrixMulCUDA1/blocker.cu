#include "MatrixMulCUDA1/blocker.cuh"

// a = mxk, b = kxn
template <int BLOCK, typename T>
__global__ void gemm_kernel1(int m, int n, int k, T *a, T *b, T *c) {
  // blockIdx control subpanel matrix

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  T *begin_a = a + bx * BLOCK * k;
  T *begin_b = b + by * BLOCK;
  T *end_a = begin_a + k;

  T sum = 0.f;

  for (T *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK * n) {
    __shared__ T ashare[BLOCK][BLOCK];
    __shared__ T bshare[BLOCK][BLOCK];

    ashare[ty][tx] = a_ptr[ty * k + tx];
    bshare[ty][tx] = b_ptr[ty * n + tx];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
      sum += ashare[ty][kk] * bshare[kk][tx];
    }
    __syncthreads();
  }

  c[(BLOCK * bx + ty) * n + BLOCK * by + tx] = sum;
}

template <size_t BLOCK, typename T>
void GEMM1(T *dA, T *dB, T *dC, int m, int n, int k) {
  /*  (BLOCK * BLOCK) threads calc (BLOCK * BLOCK) data
   *
   *  t0 t1 t2 t3 - - - - -
   *  t16 t17 t18 t19 - - - - -
   *  - - - - - - - - - - -
   *  - - - - - - - - - - -
   *  - - - - - - - - - - -
   *  - - - - - - - - - - -
   *  1.  load Gmem -> Smem each thread
   *  2.  clac FMA each thread
   */

  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  gemm_kernel1<BLOCK><<<grid, block>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM1<BLOCKSIZE, float>(float *dA, float *dB, float *dC, int m,
                                      int n, int k);
// template void GEMM1<TPB, int>(int *dA, int *dB, int *dC, int m, int n, int
// k);
