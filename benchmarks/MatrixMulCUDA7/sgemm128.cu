#include "MatrixMulCUDA7/sgemm128.cuh"

#define SMEM_LDA (128)
#define SMEM_LDB (128)

// sgemm_128x128x8
// limit maxThreadsPerBlock in 256,
// limit minBlocksPerMultiprocessor in 2 (at least 2 active warp resource)
template <typename T>
__global__ __launch_bounds__(256, 2) void gemm_kernel7(int m, int n, int k,
                                                       const T *a, const T *b,
                                                       T *c) {
  __shared__ __align__(
      16 * 1024) char smem[24 * 1024];  // 16KB shared memory for buffer

  T *ashare = reinterpret_cast<T *>(smem);              // 16K Smem for A
  T *bshare = reinterpret_cast<T *>(smem + 16 * 1024);  // 8k Smem for B

  float sum[8][8] = {0};  // do 64 FMA each thread
  T panelA[8] = {0};
  T panelB[8] = {0};

  int idx8 = threadIdx.x % 8;  // 128 ==>  8 * 16
  int idy8 = threadIdx.x / 8;

  int idx32 = threadIdx.x % 32;  // 128 ==>  32 * 4
  int idy32 = threadIdx.x / 32;

  int from_a = (blockIdx.y * 128 + idy8 * 4) * k + idx8;
  int from_b = (idy32)*n + blockIdx.x * 128 + idx32;

  /*
  *  matA(m * k) @ matb(k * n)
  *  this loop is important:
  *
  *              / -------------------------- k = 4096
  ---------------------------- /
  *              | 8  |
  *   aglobal
  --|----|----|----|----|----|----|----|----||----|----|----|----|----|
  *            16|/// |    |    |    |    |    |    |    ||/// |    |    |    |
  |
  * --|----|----|----|----|----|----|----|----||----|----|----|----|----|
  *
  *                ||
  *                ||  load && calc ---> loop(k/8)
  *                \/
  *
  *              |----|    ashare 4 * 1024 * float
  *              |warp|    bshare 2 * 1024 * flaot
  *              |----|
  *
  *                /\
  *                ||
  *                ||
  *
  *              | 32 |
  *   bglobal
  --|----|----|----|----|----|----|----|----||----|----|----|----|----|
  *            4 |/// |    |    |    |    |    |    |    ||/// |    |    |    |
  |
  * --|----|----|----|----|----|----|----|----||----|----|----|----|----|
  *              / -------------------------- k = 4096
  ---------------------------- /

  */

  for (int loop = 0; loop < k; loop += 8) {
    // part1: gmem to smem

    // load gmem to smem for ashare
    int to_a = idx8 * SMEM_LDA + idy8 * 4;  // 连续的地址不能给同一个 thread 用
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ashare[to_a + i] = a[from_a + i * k];
    }

    // load gmem to smem for bshare
    int to_b = idy32 * SMEM_LDB + idx32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      bshare[to_b + i * 32] =
          b[from_b + i * 32];  // 32 thread 合并访问。 thread i 访问  [i, i+32,
                               // i+64, i+96]
    }

    __syncthreads();
    from_a += 8;
    from_b += 8 * n;

    // part2: calculation
    // 计算 2x2 个 4x4
    int aidx0 = (threadIdx.x / 16) * 4;
    int bidx0 = (threadIdx.x % 16) * 4;

    /*
     *
     *  It must calc FMA k(8) num in each loop
     *  1. prepare 8 data to panelA (using 64 interleave)
     *  2. prepare 8 data to panelB (using 64 interleave)
     *  3. FMA
     */

#pragma unroll
    for (int subk = 0; subk < 8; ++subk) {
      T *ptrA = ashare + aidx0 + subk * SMEM_LDA;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        panelA[i] = ptrA[i];
        panelA[i + 4] = ptrA[i + 64];
      }

      const T *ptrB = bshare + bidx0 + subk * SMEM_LDB;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        panelB[i] = ptrB[i];
        panelB[i + 4] = ptrB[i + 64];
      }

#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[i][j] += (float)(panelA[i] * panelB[j]);
        }
      }
    }
    __syncthreads();
  }

  // part3: save to C
  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      c[write_offset + i * n + j] = (T)sum[i][j];
      c[write_offset + i * n + j + 64] = (T)sum[i][j + 4];
      c[write_offset + (i + 64) * n + j] = (T)sum[i + 4][j];
      c[write_offset + (i + 64) * n + j + 64] = (T)sum[i + 4][j + 4];
    }
  }
}

#undef SMEM_LDA
#undef SMEM_LDB

template <typename T>
void GEMM7(T *dA, T *dB, T *dC, int m, int n, int k) {
  constexpr int BLOCK = 128;
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  gemm_kernel7<T><<<grid, 256>>>(m, n, k, dA, dB, dC);
  cudaDeviceSynchronize();
}

template void GEMM7<float>(float *dA, float *dB, float *dC, int m, int n,
                           int k);

template void GEMM7<__half>(__half *dA, __half *dB, __half *dC, int m, int n,
                            int k);
