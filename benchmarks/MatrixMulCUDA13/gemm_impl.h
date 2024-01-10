#pragma once
#include <cuda_fp16.h>

#include <iostream>
#include <memory>
#include <sstream>

#include "MatrixMulCUDA13/common.h"
#include "MatrixMulCUDA13/gemm_base.h"
#include "MatrixMulCUDA13/global_loader.h"
#include "MatrixMulCUDA13/warp_iterator.h"

namespace mmbenchmark {

// BaseGemm[定义接口] ---> GemmImpl[继承接口, 实现接口]
//                               | ---> Gemm[协作类][计算warp次数和split-k次数]
//                                    ｜----> GlobalLoaderA[协作类][load g_A ->
//                                    s_A]
//                                     |----> GlobalLoaderB[协作类][load g_B ->
//                                     s_B]

template <typename Gemm>
__global__ void gemm_kernel(float* __restrict__ C, const float* __restrict__ A,
                            const float* __restrict__ B, int M, int N, int K) {
  Gemm{}.main_run(C, A, B, M, N, K);
}

template <int TILE_M, int TILE_N, int TILE_K, int WARP_M, int WARP_N,
          int WARP_K, int STAGES>
struct Gemm {
  // kWarpCountM = m / w_m
  static constexpr int kWarpCountM = TILE_M / WARP_M;  // 128 / 64
  static constexpr int kWarpCountN = TILE_N / WARP_N;  // 128 / 32
  static constexpr int kWarpCountK = TILE_K / WARP_K;  // 8 / 8

  static constexpr int kWarpCountMN = kWarpCountM * kWarpCountN;  // 4 * 2 = 8
  static constexpr int kWarpCount = kWarpCountMN * kWarpCountK;

  // slice = w_k
  static constexpr int SLICES = kWarpCountK;
  // slice_k = k / w_k
  static constexpr int SLICE_K = TILE_K / SLICES;  // 8 / 1

  static_assert(SLICE_K % WARP_K == 0, "infeasible sliced-k setting");

  using GlobalLoaderA = mmbenchmark::GlobalLoaderA<kWarpCountMN, TILE_M, TILE_N,
                                                   TILE_K, STAGES, SLICES>;
  using GlobalLoaderB = mmbenchmark::GlobalLoaderB<kWarpCountMN, TILE_M, TILE_N,
                                                   TILE_K, STAGES, SLICES>;

  using WarpIterA =
      mmbenchmark::WarpIteratorA<TILE_M, TILE_K, WARP_M, WARP_K,
                                 GlobalLoaderA::kSizePerStage, STAGES>;
  using WarpIterB =
      mmbenchmark::WarpIteratorB<TILE_N, TILE_K, WARP_N, WARP_K,
                                 GlobalLoaderB::kSmemPadCtaN, STAGES>;

  Array<half, 8> warp_frag_A_[2][WARP_M / OP_M];
  Array<half, 4> warp_frag_B_[2][WARP_N / OP_N];

  __device__ void warp_calc(GlobalLoaderA& iter_A, GlobalLoaderB& iter_B,
                            WarpIterA& warp_iter_A, WarpIterB& warp_iter_B,
                            float* accum, int slice_id, int& gemm_iter);
  __device__ void main_run(float* __restrict__ C, const float* __restrict__ A,
                           const float* __restrict__ B, int M, int N, int K);
  __device__ void sync_slice(int slice_id);

  template <typename T, int N>
  __device__ static void clear(T (&dst)[N]) {
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      dst[i] = T{};
    }
  }
};

template <typename TileShape, typename WarpShape, int STAGES>
struct GemmImpl : public iBaseGemm {
  static constexpr TileShape tile_shape{};
  static constexpr WarpShape warp_shape{};

  // 128, 128, 8
  // 64, 32, 1
  using GemmType = Gemm<tile_shape.m(), tile_shape.n(), tile_shape.k(),
                        warp_shape.m(), warp_shape.n(), warp_shape.k(), STAGES>;

  // decltype(&gemm_kernel<GemmType>) kernel_func_;
  // std::shared_ptr<cudaDeviceProp>     props_;
  // int                                 max_active_ctas_{};

  static constexpr int kSlices = GemmType::SLICES;
  static constexpr int kSmemSizeA =
      GemmType::GlobalLoaderA::kSmemByteSize * kSlices;
  static constexpr int kSmemSizeB =
      GemmType::GlobalLoaderB::kSmemByteSize * kSlices;
  static constexpr int kSmemSizeC =
      sizeof(float) * tile_shape.m() * tile_shape.n();
  //   //todo
  static constexpr int kSmemByteSize = kSmemSizeA + kSmemSizeB;

  // // static shared memory size of Q
  // static constexpr int kSmemSizeQ = sizeof(typename
  // GemmType::IteratorQ::Storage);

  void Launch(float* C, const float* A, const float* B, int M, int N,
              int K) override {
    constexpr int block_size = GemmType::kWarpCount * WARP_SIZE;

    std::cout << "block_size " << block_size << std::endl;
    std::cout << "kSmemByteSize " << GemmType::GlobalLoaderB::kSmemByteSize
              << std::endl;
    std::cout << "kSmemSizeC " << kSmemSizeC << std::endl;
    std::cout << "kSlices " << GemmType::SLICES << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 grid_size((M + tile_shape.m() - 1) / tile_shape.m(),
                   (N + tile_shape.n() - 1) / tile_shape.n());
    gemm_kernel<GemmType>
        <<<grid_size, block_size, kSmemByteSize, stream>>>(C, A, B, M, N, K);
  }

  void Dump(std::ostream& os) override {
    {
      os << "[Gemm] CTA shape: " << tile_shape.m() << "x" << tile_shape.n()
         << "x" << tile_shape.k() << std::endl;
      os << "[Gemm] warp shape: " << warp_shape.m() << "x" << warp_shape.n()
         << "x" << warp_shape.k() << std::endl;
      os << "[Gemm] warp count: " << GemmType::kWarpCountM << "x"
         << GemmType::kWarpCountN << "x" << GemmType::kWarpCountK << " ("
         << GemmType::kWarpCount << ")" << std::endl;
      os << std::endl;
    }

    {
      using Iter = typename GemmType::GlobalLoaderA;
      os << "[A] shape: " << Iter::kShapeM << " " << Iter::kShapeK << std::endl;
      os << "[A] warp thread arrangement: " << Iter::kWarpThreadC << " "
         << Iter::kWarpThreadS << std::endl;
      os << "[A] warp shape per access: " << Iter::kWarpAccessM << " "
         << " *** It represent the shape of warp in each global memory "
            "access.***"
         << std::endl;
      os << "[A] warp access iters: " << Iter::kWarpIterM << " "
         << Iter::kWarpIterK << std::endl;
      os << "[A] warp arrangement: " << Iter::kWarpM << " " << Iter::kWarpK
         << std::endl;
      os << "[A] iterations: " << Iter::kIterM << " " << Iter::kIterK
         << " *** It represent the iterations in load global memory. *** "
         << std::endl;
      os << "[A] iters per tile: " << Iter::kIterCount << std::endl;
      os << "[A] warp footprint: " << Iter::kWarpFootprintM << " "
         << Iter::kWarpFootprintK << std::endl;
      os << "[A] shared memory: " << Iter::kSmemByteSize << std::endl;
      os << std::endl;
    }
    {
      using Iter = typename GemmType::GlobalLoaderB;
      os << "[B] shape: "
         << "kShapeK " << Iter::kShapeK << " "
         << "kShapeN " << Iter::kShapeN << std::endl;
      os << "[B] warp thread arrangement: "
         << "kWarpThreadC " << Iter::kWarpThreadC << " "
         << "kWarpThreadS " << Iter::kWarpThreadS << std::endl;
      os << "[B] warp shape per access: "
         << "kWarpAccessK " << Iter::kWarpAccessK << " "
         << "kWarpAccessN " << Iter::kWarpAccessN << std::endl;
      os << "[B] warp access iters: "
         << "kWarpIterK " << Iter::kWarpIterK << " "
         << "kWarpIterN " << Iter::kWarpIterN << std::endl;
      os << "[B] warp arrangement: "
         << "kWarpK " << Iter::kWarpK << " "
         << "kWarpN " << Iter::kWarpN << std::endl;
      os << "[B] iterations: "
         << "kIterK " << Iter::kIterK << " "
         << "kIterN " << Iter::kIterN << std::endl;
      os << "[B] iters per tile: " << Iter::kIterCount << std::endl;
      os << "[B] kAccessSize: " << Iter::kAccessSize << std::endl;
      os << "[B] warp footprint: " << Iter::kWarpFootprintK << " "
         << Iter::kWarpFootprintN << std::endl;
      os << "[B] shared memory: " << Iter::kSmemByteSize << std::endl;
      os << std::endl;
    }
  }
};

template <typename TileShape, typename WarpShape, int Stages>
constexpr TileShape GemmImpl<TileShape, WarpShape, Stages>::tile_shape;

template <typename TileShape, typename WarpShape, int Stages>
constexpr WarpShape GemmImpl<TileShape, WarpShape, Stages>::warp_shape;

}  // namespace mmbenchmark
