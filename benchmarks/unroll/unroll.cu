#include "unroll/unroll.cuh"

// a = mxk, b = kxn
template <int BLOCK, typename T>
__global__ void gemm_kernel2(int m, int n, int k, T *a, T *b, T *c) {
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
void GEMM2(T *dA, T *dB, T*dC, int m, int n, int k) {

    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    gemm_kernel2<BLOCK><<<grid, block>>>(m, n, k, dA, dB, dC);
    cudaDeviceSynchronize();
}

template void GEMM2<TPB, float>(float *dA, float *dB, float *dC, int m, int n, int k);
// template void GEMM2<TPB, int>(int *dA, int *dB, int *dC, int m, int n, int k);
