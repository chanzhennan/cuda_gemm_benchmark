#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "MatrixMulCUDA13/common.h"
#include "MatrixMulCUDA13/czn_k16.cuh"
#include "MatrixMulCUDA13/gemm_impl.h"

template <int TILE_M, int TILE_N, int TILE_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
__device__ void
mmbenchmark::Gemm<TILE_M, TILE_N, TILE_K, WARP_M, WARP_N, WARP_K,
                  STAGES>::warp_calc(GlobalLoaderA &iter_A,
                                     GlobalLoaderB &iter_B,
                                     WarpIterA &warp_iter_A,
                                     WarpIterB &warp_iter_B, half *accum,
                                     int slice_id, int &gemm_iter) {
  constexpr int ITER_M = WARP_M / OP_M;
  constexpr int ITER_N = WARP_N / OP_N;
  constexpr int ITER_K = WARP_K / OP_K;

  constexpr int kBatchA = (GlobalLoaderA::kIterCount + ITER_K - 1) / ITER_K;
  constexpr int kBatchB = (GlobalLoaderB::kIterCount + ITER_K - 1) / ITER_K;

  auto frag_C_ptr = (Array<half, 8> *)accum;  // [ITER_N, ITER_M]

  PRAGMA_UNROLL
  for (int iter_k = 0; iter_k < ITER_K; ++iter_k) {
    warp_iter_A.load(warp_frag_A_[(iter_k + 1) % 2], (iter_k + 1) % ITER_K);
    warp_iter_B.load(warp_frag_B_[(iter_k + 1) % 2], (iter_k + 1) % ITER_K);

    auto warp_frag_A = warp_frag_A_[iter_k % 2];
    auto warp_frag_B = warp_frag_B_[iter_k % 2];

    PRAGMA_UNROLL
    for (int iter_m = 0; iter_m < 8; ++iter_m) {
      PRAGMA_UNROLL
      for (int iter_n = 0; iter_n < 8; ++iter_n) {
        auto &frag_A = warp_frag_A[iter_m];
        auto &frag_B = warp_frag_B[iter_n];
        auto &frag_C = frag_C_ptr[iter_n * 8 + iter_m];
        // frag_C += frag_A * frag_B;
      }
    }

    // if (iter_k < ITER_K - 1) {
    //   iter_A.prefetch_batch(iter_k, kBatchA, gemm_iter > 0);
    //   iter_B.prefetch_batch(iter_k, kBatchB, gemm_iter > 0);
    // }

    // if (iter_k == ITER_K - 2) {
    //   iter_A.prefetch_batch(iter_k + 1, kBatchA, gemm_iter > 0);
    //   iter_B.prefetch_batch(iter_k + 1, kBatchB, gemm_iter > 0);

    //   __pipeline_commit();
    //   __pipeline_wait_prior(STAGES - 2);
    //   sync_slice(slice_id);

    //   iter_A.next_stage();
    //   iter_B.next_stage();

    //   warp_iter_A.next_stage();
    //   warp_iter_B.next_stage();

    //   --gemm_iter;
    // }
  }
}

template <int TILE_M, int TILE_N, int TILE_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
__device__ void mmbenchmark::Gemm<TILE_M, TILE_N, TILE_K, WARP_M, WARP_N,
                                  WARP_K, STAGES>::sync_slice(int slice_id) {
  if (SLICES == 1) {
    __syncthreads();
  } else {
    constexpr int SLICE_GROUP = (SLICES + 7) / 8;
    constexpr uint32_t num_threads = kWarpCountMN * WARP_SIZE;
    const uint32_t barrier_id = slice_id / SLICE_GROUP + 1;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "n"(num_threads));
  }
}

template <int TILE_M, int TILE_N, int TILE_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
__device__ void
mmbenchmark::Gemm<TILE_M, TILE_N, TILE_K, WARP_M, WARP_N, WARP_K,
                  STAGES>::main_run(__half *__restrict__ C,
                                    const __half *__restrict__ A,
                                    const __half *__restrict__ B, int M, int N,
                                    int K) {
  __half tb_frag_C[8 * 8];
  extern __shared__ char smem[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  /*
   *
   *   step 1 calculate warp id
   *
   *
   *   | -0- N -1--|
   *
   *   |-----|-----|   -
   *   |warp0|warp4|   0
   *   |_____|_____|   |
   *   |warp1|warp5|   1
   *   |_____|_____|   |  M
   *   |warp2|warp6|   2
   *   |_____|_____|   |
   *   |warp3|warp7|   3
   *   |_____|_____|   |
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
   */

  const int warp_id_m = warp_id % kWarpCountM;
  const int warp_id_nk = warp_id / kWarpCountM;
  const int warp_id_n = warp_id_nk % kWarpCountN;
  const int warp_id_k = warp_id_nk / kWarpCountN;

  const int warp_id_mn = warp_id_n * kWarpCountM + warp_id_m;

  const int slice_id = warp_id_k;

  const int tile_kth = slice_id * SLICE_K;  // sliced-k offset
  const int tile_mth = blockIdx.x * TILE_M;
  const int tile_nth = blockIdx.y * TILE_N;

  // each slice has its own partition of smem
  __half *const tb_smem_A =
      (__half *)(smem + GlobalLoaderA::kSmemByteSize * slice_id);
  __half *const tb_smem_B =
      (__half *)(smem + GlobalLoaderA::kSmemByteSize * SLICES +
                 GlobalLoaderA::kSmemByteSize * slice_id);

  // [CTA_N / OP_N, CTA_M / OP_M, 4, WARP_SIZE], all mn fragments in CTA
  __half *const tb_smem_C = (__half *)smem;

  // clang-format off
  GlobalLoaderA iter_A{A, tb_smem_A, M, K, tile_mth, tile_kth, warp_id, lane_id};
  GlobalLoaderB iter_B{B, tb_smem_B, K, N, tile_nth, tile_kth, warp_id, lane_id};
  // clang-format on

  const int warp_x = lane_id % 8;
  const int warp_y = lane_id / 8;

  int offset_m = warp_id_m * WARP_M + warp_y * 8;
  int offset_n = warp_id_n * WARP_N + warp_x * 8;

  WarpIterA warp_iter_A(iter_A.smem_int_ptr_, warp_id_m, offset_m, 0);
  WarpIterB warp_iter_B(iter_B.smem_int_ptr_, warp_id_n, offset_n, 0);

  int gemm_iter = (K + TILE_K - 1) / TILE_K;

  for (int stage = 0; stage < STAGES - 1; ++stage, --gemm_iter) {
    iter_A.prefetch_stage(gemm_iter > 0);
    iter_B.prefetch_stage(gemm_iter > 0);
    __pipeline_commit();
  }
  // float *tmpb = (float *)iter_B.smem_;
  // if (threadIdx.x < 64 && blockIdx.x == 0 && blockIdx.y == 0) {
  //   printf("tmpb[%d] = %f gloab[%d] = %f\n", threadIdx.x, tmpb[threadIdx.x],
  //          threadIdx.x, B[threadIdx.x]);
  // }

  if (threadIdx.x % 32 == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("tmpb[%d] offset_m = %d \n", threadIdx.x, offset_m);
    // printf("tmpb[%d] offset_k_ = %d offset_n =%d ldsm_group_id_ = %d
    // warp_id_n = %d warp_id_k = %d warp_id_m = %d warp_id_mn=%d warpId =
    // %d\n", threadIdx.x, warp_iter_B.offset_k_, warp_iter_B.offset_n_,
    // warp_iter_B.ldsm_group_id_, warp_id_n, warp_id_k, warp_id_m, warp_id_mn,
    // warp_id);
  }

  // /************ above correct ***/
  //   clear(tb_frag_C);

  //   __pipeline_wait_prior(STAGES - 2);
  //   sync_slice(slice_id);

  //   if (threadIdx.x ==0 && blockIdx.x == 0 && blockIdx.y == 0) {
  //     printf("WARP_N = %d WARP_M = %d WARP_K = %d \n", WARP_N, WARP_M,
  //     WARP_K);
  //   }

  //   warp_iter_A.load(warp_frag_A_[0], 0);
  //   warp_iter_B.load(warp_frag_B_[0], 0);

  // PRAGMA_NO_UNROLL
  // for (; gemm_iter > -STAGES + 1;) {
  //   warp_calc(iter_A, iter_B, warp_iter_A, warp_iter_B, tb_frag_C, slice_id,
  //             gemm_iter);
  // }

  // __pipeline_commit();
  // __pipeline_wait_prior(0);
  // __syncthreads();
}

template <typename T>
void GEMM13(T *dA, T *dB, T *dC, int m, int n, int k) {
  auto gemm = new mmbenchmark::GemmImpl<mmbenchmark::Shape<128, 128, 8>,
                                        mmbenchmark::Shape<32, 64, 8>, 2>{};
  std::ostream &outputStream = std::cout;
  gemm->Dump(outputStream);
  gemm->Launch(dB, dC, dA, m, n, k);
}

// template void GEMM13<float>(float *dA, float *dB, float *dC, int m, int n,
//                             int k);
template void GEMM13<__half>(__half *dA, __half *dB, __half *dC, int m, int n,
                             int k);
