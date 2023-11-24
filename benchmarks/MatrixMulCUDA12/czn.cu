#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "MatrixMulCUDA11/yhs.cuh"

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

__device__ __forceinline__ void ldg32_nc(float &reg, const void *ptr,
                                         bool guard) {
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
      " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
      " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
      : "=f"(reg)
      : "l"(ptr), "r"((int)guard));
}

__device__ __forceinline__ void ldg32_nc_0(float &reg, const void *ptr,
                                           bool guard) {
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
      " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
      " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
      : "=f"(reg)
      : "l"(ptr), "r"((int)guard));
}

__device__ __forceinline__ void ldg128(float &reg0, float &reg1, float &reg2,
                                       float &reg3, const void *ptr,
                                       bool guard) {
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "l"(ptr), "r"((int)guard));
}

__device__ __forceinline__ void stg32(const float &reg, void *ptr, bool guard) {
  asm volatile(
      "{.reg .pred p;\n"
      " setp.ne.b32 p, %2, 0;\n"
      " @p st.global.f32 [%0], %1;}\n"
      :
      : "l"(ptr), "f"(reg), "r"((int)guard));
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2,
                                       float &reg3, const uint32_t &addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "r"(addr));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr) {
  asm volatile("st.shared.f32 [%0], %1;\n" : : "r"(addr), "f"(reg));
}

__device__ __forceinline__ void sts128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const uint32_t &addr) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

struct StgFrag {
  float data[4][4];

  __device__ __forceinline__ StgFrag(const float (&C_frag)[8][8], int tile_x,
                                     int tile_y) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
      }
    }
  }
};

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag
 * --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---| 1|///|
 * |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|-- |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 | |
 *    |---|--   |---|---|---|---|---|---|---|---||---| | |   |     |t1 |t3 |t5
 *              |t7 |t9 |t11|t13|t15||                               | 16|---|
 * |---|---|---|---|---|---|---|---||                               | |   |
 * |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---|| | |   |
 * |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 | |
 *    |---|     |---|                           ||---| | |   |     | || |
 *    |---|     |                               || | |   |     | || |
 *    |---|     |                               || | |   |     | || |
 *    |---| |-------------------------------||-------------------------------|
 *
 */

__device__ void debugShd(float *A_smem, float *B_smem) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("\nkernel A\n ");
    for (int i = 0; i < 132; i++) {
      printf("%.2f ", A_smem[i]);
    }
    printf("\n ");
    printf("\nkernel B\n ");
    for (int i = 0; i < 132; i++) {
      printf("%.2f ", B_smem[i]);
    }
    printf("\n ");
  }
}

__device__ void debugShd2(float *A_smem, float *B_smem) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("\nkernel A\n ");
    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 8; i++) {
        printf("%.2f ", A_smem[i + j * 132]);
      }
      printf("\n ");
    }
    printf("\n ");

    printf("\nkernel B\n ");
    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 8; i++) {
        printf("%.2f ", B_smem[i + j * 128]);
      }
      printf("\n ");
    }
    printf("\n ");
  }
}

__device__ void debugReg(float **A_frag, float **B_frag, uint32_t A_lds_addr,
                         uint32_t B_lds_addr, float *B_smem, float *A_smem) {
  int k_frag = 1;
  int next_frag = (k_frag + 1) % 2;

  lds128(A_frag[next_frag][0], A_frag[next_frag][1], A_frag[next_frag][2],
         A_frag[next_frag][3],
         A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
  lds128(A_frag[next_frag][4], A_frag[next_frag][5], A_frag[next_frag][6],
         A_frag[next_frag][7],
         A_lds_addr + ((k_frag + 1) % 8 * 132 + 4) * sizeof(float));
  lds128(B_frag[next_frag][0], B_frag[next_frag][1], B_frag[next_frag][2],
         B_frag[next_frag][3],
         B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
  lds128(B_frag[next_frag][4], B_frag[next_frag][5], B_frag[next_frag][6],
         B_frag[next_frag][7],
         B_lds_addr + ((k_frag + 1) % 8 * 128 + 4) * sizeof(float));

  __syncthreads();

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("\n1111kernel A reg\n ");
    for (int i = 0; i < 8; i++) {
      printf("%.2f ", B_frag[next_frag][i]);
    }

    for (int j = 0; j < 8; j++) {
      printf("xx %.2f ", B_smem[128 + j]);
    }

    printf("\n ");
  }
}

// A(k * n) B(m * k)
__global__ __launch_bounds__(256, 2) void sgemm_128x128x8_kernel_my(
    const float *A, const float *B, float *C, uint32_t m, uint32_t n,
    uint32_t k,
    uint32_t A_ldg_step,    // k * sizeof(float)
    uint32_t B_ldg_step) {  // n * sizeof(float) * 8
  /*
   * matrix A & B thread block tile shared memory (double buffer)
   * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
   * matrix B: 128 * 8 * 4Byte/item * double buffer = 8KB
   *
   * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
   * and 16KB aligned, B_smem should be 8KB aligned, then the double buffer
   * can be switched by only 1 xor instruction:
   *     (uint32_t &)A_smem ^= 0x2000;
   *     (uint32_t &)B_smem ^= 0x1000;
   */
  __shared__ __align__(16 * 1024) char smem[24 * 1024];
  float *A_smem = reinterpret_cast<float *>(smem);
  float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);

  // A, B and C register fragment
  float A_frag[2][8];
  float B_frag[2][8];
  float C_frag[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      C_frag[i][j] = 0;
    }
  }

  const char *A_gmem =
      (const char *)(A + k * (blockIdx.y * 128 + (threadIdx.x / 8) * 4) +
                     (threadIdx.x % 8));
  const char *B_gmem =
      (const char *)(B + (threadIdx.x / 32) * n + blockIdx.x * 128 +
                     (threadIdx.x % 32) * 4);

  int warp_xth = threadIdx.x / 32;
  int thread_xth = threadIdx.x % 32;

  int thread_xth_in_a = thread_xth / 8;
  int thread_xth_in_b = thread_xth % 8;

  // 4x8 threads each warp for FFMA
  const uint32_t mma_tid_x = (thread_xth / 2) % 8;
  const uint32_t mma_tid_y = (thread_xth / 16) * 2 + (thread_xth % 2);

  // shared ptr
  // todo explain this formula
  uint32_t A_sts_addr =
      smem_u32addr(A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
  uint32_t B_sts_addr =
      smem_u32addr(B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32) * 4);

  // uint32_t A_lds_addr = smem_u32addr(A_smem + (warp_xth / 2) * 32 +
  // thread_xth_in_a * 8); uint32_t B_lds_addr = smem_u32addr(B_smem + (warp_xth
  // % 2) * 64 + thread_xth_in_b * 8);

  uint32_t A_lds_addr =
      smem_u32addr(A_smem + (warp_xth / 2) * 32 + mma_tid_y * 4);
  uint32_t B_lds_addr =
      smem_u32addr(B_smem + (warp_xth % 2) * 64 + mma_tid_x * 4);

  // (lane_id / 16) * 2 + (lane_id % 2)

  uint32_t A_ldg_guard = 0;
#pragma unroll
  int m_idx = blockIdx.y * 128 + (threadIdx.x / 8) * 4;
  for (int i = 0; i < 4; ++i) {
    if (m_idx + i < m) {
      A_ldg_guard |= (1u << i);
    }
  }

  uint32_t B_ldg_guard = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int n_idx = blockIdx.x * 128 + (threadIdx.x % 32) * 4 + i;
    if (n_idx < n) {
      B_ldg_guard |= (1u << i);
    }
  }

  __syncthreads();

  uint32_t k_tiles = (k + 7) / 8 - 1;
  uint32_t first_k_tile = k - k_tiles * 8;
  float A_ldg_reg[4];
  float B_ldg_reg[4];

  //  gmem(1) -> smem(1)
  {
    ldg128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3], B_gmem,
           B_ldg_guard);
    sts128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3], B_sts_addr);

    for (int i = 0; i < 4; i++) {
      bool guard = (A_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < k;
      ldg32_nc_0(A_ldg_reg[i], A_gmem + i * k * sizeof(float), guard);
    }
    sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3], A_sts_addr);

    __syncthreads();

    // switch double buffer
    A_sts_addr ^= 0x2000;
    B_sts_addr ^= 0x1000;

    // ldg pointer for next tile
    A_gmem += first_k_tile * sizeof(float);
    B_gmem += n * first_k_tile * sizeof(float);
  }

  // smem(1) -> rmem(1)
  lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
  lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
         A_lds_addr + 4 * sizeof(float));

  lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
  lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
         B_lds_addr + 4 * sizeof(float));
  __syncthreads();

  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
  // {
  //   printf("k_tiles = %d\n", k_tiles);
  // }

  int next_frag = 0;
  int this_frag = 0;

  for (; k_tiles > 0; --k_tiles) {
#pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
      next_frag = (k_frag + 1) % 2;
      this_frag = (k_frag) % 2;

      // 1. TReg(2) -> shared(2)
      // 2. synch()
      // 3. rotated buff
      // 4. update global ptr
      if (k_frag == 7) {
        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);
        sts128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3],
               B_sts_addr);
        __syncthreads();

        A_lds_addr ^= 0x2000;
        B_lds_addr ^= 0x1000;
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x1000;

        A_gmem += 8 * sizeof(float);
        B_gmem += 8 * n * sizeof(float);
      }

      // shared(2) -> reg(2)
      lds128(A_frag[next_frag][0], A_frag[next_frag][1], A_frag[next_frag][2],
             A_frag[next_frag][3],
             A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
      lds128(A_frag[next_frag][4], A_frag[next_frag][5], A_frag[next_frag][6],
             A_frag[next_frag][7],
             A_lds_addr + ((k_frag + 1) % 8 * 132 + 4) * sizeof(float));

      lds128(B_frag[next_frag][0], B_frag[next_frag][1], B_frag[next_frag][2],
             B_frag[next_frag][3],
             B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
      lds128(B_frag[next_frag][4], B_frag[next_frag][5], B_frag[next_frag][6],
             B_frag[next_frag][7],
             B_lds_addr + ((k_frag + 1) % 8 * 128 + 4) * sizeof(float));

      if (k_frag == 0) {
        // Gmem(2) -> TReg(2)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          bool guard = (A_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < k;
          ldg32_nc_0(A_ldg_reg[i], A_gmem + i * k * sizeof(float), guard);
        }

        ldg128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3], B_gmem,
               B_ldg_guard);
      }

      // FFMA loop
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          C_frag[i][j] += A_frag[this_frag][i] * B_frag[this_frag][j];
        }
      }
    }
  }

  // debugShd(A_smem, B_smem);
  // debugReg((float**)A_frag, (float**)B_frag, A_lds_addr, B_lds_addr);
  // float *tmp_A = (float*)__cvta_shared_to_generic(A_lds_addr);
  // float *tmp_B = (float*)__cvta_shared_to_generic(B_lds_addr);
  // debugShd2(tmp_A, tmp_B);

  lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
  lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
         A_lds_addr + 4 * sizeof(float));
  lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
  lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
         B_lds_addr + 4 * sizeof(float));

  for (int k_frag = 0; k_frag < 8; k_frag++) {
    next_frag = (k_frag + 1) % 2;
    this_frag = (k_frag) % 2;

    if (k_frag < 7) {
      lds128(A_frag[next_frag][0], A_frag[next_frag][1], A_frag[next_frag][2],
             A_frag[next_frag][3],
             A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));

      lds128(A_frag[next_frag][4], A_frag[next_frag][5], A_frag[next_frag][6],
             A_frag[next_frag][7],
             A_lds_addr + ((k_frag + 1) % 8 * 132 + 4) * sizeof(float));

      lds128(B_frag[next_frag][0], B_frag[next_frag][1], B_frag[next_frag][2],
             B_frag[next_frag][3],
             B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
      lds128(B_frag[next_frag][4], B_frag[next_frag][5], B_frag[next_frag][6],
             B_frag[next_frag][7],
             B_lds_addr + ((k_frag + 1) % 8 * 128 + 4) * sizeof(float));
    }
    // calc
    // FFMA loop
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        C_frag[i][j] += A_frag[this_frag][i] * B_frag[this_frag][j];
      }
    }
  }

  int xx = (threadIdx.x % 16);
  int yy = (threadIdx.x / 16);

  float *global_c = C + blockIdx.x * 128 + blockIdx.y * 128 * n;
  float *tile_c;
#pragma unroll
  for (int i = 0; i < 8; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      global_c[xx * 8 + j + (yy * 8 + i) * n] = C_frag[i][j];
    }
  }

  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
  // {
  //   for (int i = 0; i < 8; i++){
  //   for (int j = 0; j < 8; j++){

  //     printf("%.2f ", C_frag[i][j]);
  //   }
  //   printf("\n ");

  // }
  // }
}

// refer to  https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs
template <typename T>
void GEMM12(T *dA, T *dB, T *dC, int m, int n, int k) {
  dim3 grid((n + 127) / 128, (m + 127) / 128);
  sgemm_128x128x8_kernel_my<<<grid, 256>>>(
      dA, dB, dC, m, n, k, k * sizeof(float), n * sizeof(float) * 8);
  cudaDeviceSynchronize();
}

template void GEMM12<float>(float *dA, float *dB, float *dC, int m, int n,
                            int k);
