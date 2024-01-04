#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "MatrixMulCUDA13/common.h"
#include "MatrixMulCUDA13/czn_k16.cuh"
#include "MatrixMulCUDA13/gemm_impl.h"

static constexpr int OP_M = 16;
static constexpr int OP_N = 8;
static constexpr int OP_K = 16;

template <int TILE_M, int TILE_N, int TILE_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
__device__ void
mmbenchmark::Gemm<TILE_M, TILE_N, TILE_K, WARP_M, WARP_N, WARP_K, STAGES>::main_run(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int M, int N, int K) {
  float tb_frag_C[(WARP_N / OP_N) * (WARP_M / OP_M) * 4];

  extern __shared__ char smem[];

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

  const int tile_kth = slice_id * SLICE_K;  // sliced-k offset
  const int tile_mth = blockIdx.x * TILE_M;
  const int tile_nth = blockIdx.y * TILE_N;

  // if (threadIdx.x < 64 && blockIdx.y == 0 && blockIdx.x == 0) {
  //   printf("XXXX WARP_N = %d WARP_M = %d\n", WARP_N, WARP_M);
  //   printf("XXXX slice_id = %d SLICE_K = %d \n", slice_id, SLICE_K);
  //   printf("XXXX tile_k = %d m_th_tile = %d n_th_tile = %d warp_id_mn =
  //   %d\n", tile_kth,
  //          tile_mth, tile_nth, warp_id_mn);
  // }

  // each slice has its own partition of smem
  float *const tb_smem_A =
      (float *)(smem + GlobalLoaderA::kSmemByteSize * slice_id);
  float *const tb_smem_B =
      (float *)(smem + GlobalLoaderA::kSmemByteSize * SLICES +
                GlobalLoaderA::kSmemByteSize * slice_id);

  // [CTA_N / OP_N, CTA_M / OP_M, 4, WARP_SIZE], all mn fragments in CTA
  float *const tb_smem_C = (float *)smem;

  // clang-format off
  GlobalLoaderA iter_A{A, tb_smem_A, M, K, tile_mth, tile_kth, warp_id_mn, lane_id};
  GlobalLoaderB iter_B{B, tb_smem_B, K, N, tile_nth, tile_kth, warp_id_mn, lane_id};
  // clang-format on

  int gemm_iter = (K + TILE_K - 1) / TILE_K;

  for (int stage = 0; stage < STAGES - 1; ++stage, --gemm_iter) {
    // iter_A.prefetch_stage(gemm_iter > 0);
    iter_B.prefetch_stage(gemm_iter > 0);
    __pipeline_commit();
  }
  float *tmpb = (float *)iter_B.smem_;
  if (threadIdx.x < 64 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("tmpb[%d] = %f gloab[%d] = %f\n", threadIdx.x, tmpb[threadIdx.x],
           threadIdx.x, B[threadIdx.x]);
  }
}

template <typename T>
void GEMM13(T *dA, T *dB, T *dC, int m, int n, int k) {
  // auto gemm = new mmbenchmark::GemmKernel{};
  auto gemm = new mmbenchmark::GemmImpl<mmbenchmark::Shape<128, 128, 8>,
                                          mmbenchmark::Shape<32, 64, 8>, 3>{};
  std::ostream &outputStream = std::cout;
  gemm->Dump(outputStream);
  gemm->Launch(dB, dC, dA, m, n, k);
}

template void GEMM13<float>(float *dA, float *dB, float *dC, int m, int n,
                            int k);
