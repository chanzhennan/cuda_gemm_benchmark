#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "MatrixMulCUDA13/common.h"
#include "MatrixMulCUDA13/czn_k16.cuh"
#include "MatrixMulCUDA13/gemm_kernel.h"

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
    for (int j = 0; j < 16; j++) {
      for (int i = 0; i < 8; i++) {
        printf("%.2f ", A_smem[i + j * 132]);
      }
      printf("\n ");
    }
    printf("\n ");

    printf("\nkernel B\n ");
    for (int j = 0; j < 16; j++) {
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

// // A(k * n) B(m * k)
// __global__ __launch_bounds__(256, 2) void sgemm_128x128x16_kernel_temp(
//     const float *A, const float *B, float *C, uint32_t m, uint32_t n,
//     uint32_t k,
//     uint32_t A_ldg_step,    // k * sizeof(float)
//     uint32_t B_ldg_step) {  // n * sizeof(float) * 8

//   /*
//   *   16 * 128 * sizeof(float) = 8K
//   *   double buffer = 16K, A + B = 32k
//   */
//   __shared__ __align__(32 * 1024) char smem[48 * 1024];
//   float *A_smem = reinterpret_cast<float *>(smem);
//   float *B_smem = reinterpret_cast<float *>(smem + 32 * 1024);

//   // A, B and C register fragment
//   float A_frag[2][8];
//   float B_frag[2][8];
//   float C_frag[8][8];
// #pragma unroll
//   for (int i = 0; i < 8; ++i) {
// #pragma unroll
//     for (int j = 0; j < 8; ++j) {
//       C_frag[i][j] = 0;
//     }
//   }

//   const char *A_gmem =
//       (const char *)(A + k * (blockIdx.y * 128 + (threadIdx.x / 8) * 4) +
//                      (threadIdx.x % 8));
//   const char *B_gmem =
//       (const char *)(B + (threadIdx.x / 32) * n + blockIdx.x * 128 +
//                      (threadIdx.x % 32) * 4);

//   const char *A_gmem_8 =
//       (const char *)(A + k * (blockIdx.y * 128 + (threadIdx.x / 8) * 4) +
//                      (threadIdx.x % 8) + 8);
//   const char *B_gmem_8 =
//       (const char *)(B + (threadIdx.x / 32) * n + blockIdx.x * 128 +
//                      (threadIdx.x % 32) * 4 + n * 8);

//   int warp_xth = threadIdx.x / 32;
//   int thread_xth = threadIdx.x % 32;

//   int thread_xth_in_a = thread_xth / 8;
//   int thread_xth_in_b = thread_xth % 8;

//   // 4x8 threads each warp for FFMA
//   const uint32_t mma_tid_x = (thread_xth / 2) % 8;
//   const uint32_t mma_tid_y = (thread_xth / 16) * 2 + (thread_xth % 2);

//   // shared ptr
//   // todo explain this formula
//   uint32_t A_sts_addr =
//       smem_u32addr(A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
//   uint32_t B_sts_addr =
//       smem_u32addr(B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32) *
//       4);

//   uint32_t A_sts_addr2 =
//       smem_u32addr(A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4 +
//       8 * 132);
//   uint32_t B_sts_addr2 =
//       smem_u32addr(B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32) * 4
//       + 8 * 128);

//   // uint32_t A_lds_addr = smem_u32addr(A_smem + (warp_xth / 2) * 32 +
//   // thread_xth_in_a * 8); uint32_t B_lds_addr = smem_u32addr(B_smem +
//   (warp_xth
//   // % 2) * 64 + thread_xth_in_b * 8);

//   uint32_t A_lds_addr =
//       smem_u32addr(A_smem + (warp_xth / 2) * 32 + mma_tid_y * 4);
//   uint32_t B_lds_addr =
//       smem_u32addr(B_smem + (warp_xth % 2) * 64 + mma_tid_x * 4);

//   // (lane_id / 16) * 2 + (lane_id % 2)

//   uint32_t A_ldg_guard = 0;
// #pragma unroll
//   int m_idx = blockIdx.y * 128 + (threadIdx.x / 8) * 4;
//   for (int i = 0; i < 4; ++i) {
//     if (m_idx + i < m) {
//       A_ldg_guard |= (1u << i);
//     }
//   }

//   uint32_t B_ldg_guard = 0;
// #pragma unroll
//   for (int i = 0; i < 4; ++i) {
//     int n_idx = blockIdx.x * 128 + (threadIdx.x % 32) * 4 + i;
//     if (n_idx < n) {
//       B_ldg_guard |= (1u << i);
//     }
//   }

//   __syncthreads();

//   uint32_t k_tiles = (k + 15) / 16 - 1;
//   uint32_t first_k_tile = k - k_tiles * 16;
//   float A_ldg_reg[4];
//   float A_ldg_reg2[4];
//   float B_ldg_reg[4];
//   float B_ldg_reg2[4];

//   // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
//   //   printf("k_tiles = %d first_k_tile = %d \n", k_tiles, first_k_tile);
//   // }

//   //  gmem(1) -> smem(1)
//   {
//     // first 8
//     ldg128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3], B_gmem,
//            B_ldg_guard);
//     sts128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3],
//     B_sts_addr);

//     for (int i = 0; i < 4; i++) {
//       bool guard = (A_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < k;
//       ldg32_nc_0(A_ldg_reg[i], A_gmem + i * k * sizeof(float), guard);
//     }
//     sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
//     A_sts_addr);

//     // second 8
//     ldg128(B_ldg_reg2[0], B_ldg_reg2[1], B_ldg_reg2[2], B_ldg_reg2[3],
//     B_gmem_8,
//            B_ldg_guard);
//     sts128(B_ldg_reg2[0], B_ldg_reg2[1], B_ldg_reg2[2], B_ldg_reg2[3],
//     B_sts_addr2);

//     for (int i = 0; i < 4; i++) {
//       bool guard = (A_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < k;
//       ldg32_nc_0(A_ldg_reg2[i], A_gmem_8 + i * k * sizeof(float), guard);
//     }
//     sts128(A_ldg_reg2[0], A_ldg_reg2[1], A_ldg_reg2[2], A_ldg_reg2[3],
//     A_sts_addr2);

//     __syncthreads();

//     // switch double buffer
//     A_sts_addr ^= 0x4000;
//     A_sts_addr2 ^= 0x4000;

//     B_sts_addr ^= 0x2000;
//     B_sts_addr2 ^= 0x2000;

//     // ldg pointer for next tile
//     A_gmem += first_k_tile * sizeof(float);
//     B_gmem += n * first_k_tile * sizeof(float);
//     A_gmem_8 += first_k_tile * sizeof(float);
//     B_gmem_8 +=  n * first_k_tile * sizeof(float);
//   }
// //check

//   // float* shared_a = A_smem;
//   // float* shared_b = B_smem;
//   // debugShd2(A_smem, B_smem);

//   // smem(1) -> rmem(1)
//   lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
//   lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
//          A_lds_addr + 4 * sizeof(float));

//   lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
//   lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
//          B_lds_addr + 4 * sizeof(float));

//   __syncthreads();

//   int next_frag = 0;
//   int this_frag = 0;

//   for (; k_tiles > 0; --k_tiles) {
// #pragma unroll
//     for (int k_frag = 0; k_frag < 16; ++k_frag) {
//       next_frag = (k_frag + 1) % 2;
//       this_frag = (k_frag) % 2;

//       // 1. TReg(2) -> shared(2)
//       // 2. synch()
//       // 3. rotated buff
//       // 4. update global ptr
//       if (k_frag == 15) {
//         sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2],
//         A_ldg_reg[3],A_sts_addr); sts128(B_ldg_reg[0], B_ldg_reg[1],
//         B_ldg_reg[2], B_ldg_reg[3],B_sts_addr);

//         sts128(A_ldg_reg2[0], A_ldg_reg2[1], A_ldg_reg2[2], A_ldg_reg2[3],
//         A_sts_addr2); sts128(B_ldg_reg2[0], B_ldg_reg2[1], B_ldg_reg2[2],
//         B_ldg_reg2[3], B_sts_addr2);

//         __syncthreads();

//         A_lds_addr ^= 0x4000;
//         B_lds_addr ^= 0x2000;

//         A_sts_addr ^= 0x4000;
//         A_sts_addr2 ^= 0x4000;

//         B_sts_addr ^= 0x2000;
//         B_sts_addr2 ^= 0x2000;

//         A_gmem += 16 * sizeof(float);
//         B_gmem += 16 * n * sizeof(float);
//         A_gmem_8 += 16 * sizeof(float);
//         B_gmem_8 += 16 * n * sizeof(float);
//       }

//       // shared(2) -> reg(2)
//       lds128(A_frag[next_frag][0], A_frag[next_frag][1],
//       A_frag[next_frag][2],
//              A_frag[next_frag][3],
//              A_lds_addr + (k_frag + 1) % 16 * 132 * sizeof(float));
//       lds128(A_frag[next_frag][4], A_frag[next_frag][5],
//       A_frag[next_frag][6],
//              A_frag[next_frag][7],
//              A_lds_addr + ((k_frag + 1) % 16 * 132 + 4) * sizeof(float));

//       lds128(B_frag[next_frag][0], B_frag[next_frag][1],
//       B_frag[next_frag][2],
//              B_frag[next_frag][3],
//              B_lds_addr + (k_frag + 1) % 16 * 128 * sizeof(float));
//       lds128(B_frag[next_frag][4], B_frag[next_frag][5],
//       B_frag[next_frag][6],
//              B_frag[next_frag][7],
//              B_lds_addr + ((k_frag + 1) % 16 * 128 + 4) * sizeof(float));

//       if (k_frag == 0) {
//         // Gmem(2) -> TReg(2)
// #pragma unroll
//         for (int i = 0; i < 4; i++) {
//           ldg32_nc(A_ldg_reg[i], A_gmem + i * k * sizeof(float),
//                    (A_ldg_guard & (1u << i)) != 0);
//         }

// #pragma unroll
//         for (int i = 0; i < 4; i++) {
//           ldg32_nc(A_ldg_reg2[i], A_gmem_8 + i * k * sizeof(float),
//                    (A_ldg_guard & (1u << i)) != 0);
//         }

//         ldg128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3],
//         B_gmem,
//                B_ldg_guard);

//         ldg128(B_ldg_reg2[0], B_ldg_reg2[1], B_ldg_reg2[2], B_ldg_reg2[3],
//         B_gmem_8,
//            B_ldg_guard);
//       }

//       // FFMA loop
// #pragma unroll
//       for (int i = 0; i < 8; ++i) {
// #pragma unroll
//         for (int j = 0; j < 8; ++j) {
//           C_frag[i][j] += A_frag[this_frag][i] * B_frag[this_frag][j];
//         }
//       }
//     }
//   }

//   // debugShd(A_smem, B_smem);
//   // debugReg((float**)A_frag, (float**)B_frag, A_lds_addr, B_lds_addr);
//   // float *tmp_A = (float*)__cvta_shared_to_generic(A_lds_addr);
//   // float *tmp_B = (float*)__cvta_shared_to_generic(B_lds_addr);
//   // debugShd2(tmp_A, tmp_B);

//   lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
//   lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
//          A_lds_addr + 4 * sizeof(float));
//   lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
//   lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
//          B_lds_addr + 4 * sizeof(float));

// #pragma unroll
//   for (int k_frag = 0; k_frag < 16; k_frag++) {
//     next_frag = (k_frag + 1) % 2;
//     this_frag = (k_frag) % 2;

//     if (k_frag < 15) {
//       lds128(A_frag[next_frag][0], A_frag[next_frag][1],
//       A_frag[next_frag][2],
//              A_frag[next_frag][3],
//              A_lds_addr + (k_frag + 1) % 16 * 132 * sizeof(float));

//       lds128(A_frag[next_frag][4], A_frag[next_frag][5],
//       A_frag[next_frag][6],
//              A_frag[next_frag][7],
//              A_lds_addr + ((k_frag + 1) % 16 * 132 + 4) * sizeof(float));

//       lds128(B_frag[next_frag][0], B_frag[next_frag][1],
//       B_frag[next_frag][2],
//              B_frag[next_frag][3],
//              B_lds_addr + (k_frag + 1) % 16 * 128 * sizeof(float));
//       lds128(B_frag[next_frag][4], B_frag[next_frag][5],
//       B_frag[next_frag][6],
//              B_frag[next_frag][7],
//              B_lds_addr + ((k_frag + 1) % 16 * 128 + 4) * sizeof(float));
//     }

//   // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
//   //   for (int i = 0; i < 8; i++)
//   //   {
//   //     printf("%.2f ", A_frag[this_frag][i]);
//   //   }
//   //   printf("\n ");
//   // }

//     // calc
//     // FFMA loop
// #pragma unroll
//     for (int i = 0; i < 8; ++i) {
// #pragma unroll
//       for (int j = 0; j < 8; ++j) {
//         C_frag[i][j] += A_frag[this_frag][i] * B_frag[this_frag][j];
//       }
//     }
//   }

//   if (C[threadIdx.x] < 0)
//   {
//     C[threadIdx.x] =  C_frag[0][0];
//   }

// }

static constexpr int OP_M = 16;
static constexpr int OP_N = 8;
static constexpr int OP_K = 16;

template <int TILE_M, int TILE_N, int TILE_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
__device__ void
mmbenchmark::Gemm<TILE_M, TILE_N, TILE_K, WARP_M, WARP_N, WARP_K, STAGES>::run(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int M, int N, int K) {
  float tb_frag_C[(WARP_N / OP_N) * (WARP_M / OP_M) * 4];

  extern __shared__ uint8_t smem[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  /*
   *
   *   step 1 calculate warp id
   *
   *   |-----|-----|-----|-----|
   *   |warp0|warp1|     |     |
   *   |_____|_____|_____|_____|
   *   |     |     |     |     |
   *   |_____|_____|_____|_____|
   *
   *
   *   step2 divide into warp_id_m warp_id_n
   *
   *   warp_id_nk
   *   |
   *   1                                         -->   warp_id_k
   *   |                                               0
   *   0                                         -->   |
   *   -- 0 --- 1 --- 2 --- 3 ---  warp_id_m           -- 0 -- 1 -- warp_id_n
   *
   *
   *
   *
   *
   */

  const int warp_id_m = warp_id % kWarpCountM;
  const int warp_id_nk = warp_id / kWarpCountM;
  const int warp_id_n = warp_id_nk % kWarpCountN;
  const int warp_id_k = warp_id_nk / kWarpCountN;

  const int warp_id_mn = warp_id_n * kWarpCountM + warp_id_m;

  const int slice_id = warp_id_k;

  const int cta_k = slice_id * SLICE_K;  // sliced-k offset
  const int m_th_tile = blockIdx.x * TILE_M;
  const int n_th_tile = blockIdx.y * TILE_N;

  // each slice has its own partition of smem
  float *const tb_smem_A =
      (float *)(smem + GlobalLoaderA::kSmemByteSize * slice_id);
  float *const tb_smem_B =
      (float *)(smem + GlobalLoaderA::kSmemByteSize * SLICES +
                GlobalLoaderA::kSmemByteSize * slice_id);

  // [CTA_N / OP_N, CTA_M / OP_M, 4, WARP_SIZE], all mn fragments in CTA
  float *const tb_smem_C = (float *)smem;

  if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 0) {
    printf("WARP_N = %d OP_N = %d WARP_M = %d OP_M = %d\n", WARP_N, OP_N,
           WARP_M, OP_M);
    printf("slice_id = %d SLICE_K = %d \n", slice_id, SLICE_K);
    printf("cta_k = %d m_th_tile = %d n_th_tile = %d warp_id_mn = %d\n", cta_k,
           m_th_tile, n_th_tile, warp_id_mn);
  }

  GlobalLoaderA iter_A{A,         tb_smem_A, M,          K,
                       m_th_tile, cta_k,     warp_id_mn, lane_id};
  GlobalLoaderB iter_B{B,         tb_smem_B, K,          N,
                       n_th_tile, cta_k,     warp_id_mn, lane_id};
}

template <typename T>
void GEMM13(T *dA, T *dB, T *dC, int m, int n, int k) {
  // auto gemm = new mmbenchmark::GemmKernel{};
  auto gemm = new mmbenchmark::GemmKernel<mmbenchmark::Shape<128, 128, 8>,
                                          mmbenchmark::Shape<64, 32, 1>, 3>{};
  std::ostream &outputStream = std::cout;
  gemm->Dump(outputStream);
  gemm->Launch(dB, dC, dA, m, n, k);
}

template void GEMM13<float>(float *dA, float *dB, float *dC, int m, int n,
                            int k);
