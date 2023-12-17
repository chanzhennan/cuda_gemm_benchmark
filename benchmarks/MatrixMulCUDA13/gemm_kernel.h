#pragma once
#include <cuda_fp16.h>

#include <iostream>
#include <memory>
#include <sstream>

#include "MatrixMulCUDA13/common.h"
#include "MatrixMulCUDA13/global_loader.h"
#include "MatrixMulCUDA13/virtual_kernel.h"

namespace mmbenchmark {

template <typename Gemm>
__global__ void gemm_kernel(float* __restrict__ C, const float* __restrict__ A,
                            const float* __restrict__ B, int M, int N, int K) {
  Gemm{}.run(C, A, B, M, N, K);
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

  __device__ void run(float* __restrict__ C, const float* __restrict__ A,
                      const float* __restrict__ B, int M, int N, int K);
};

template <typename TileShape, typename WarpShape, int STAGES>
struct GemmKernel : public CoreKernel {
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

    //  {
    //    using Iter = typename GemmType::GlobalLoaderA;
    //    os << "[A] shape: " << Iter::kShapeM << " " << Iter::kShapeK <<
    //    std::endl; os << "[A] warp thread arrangement: " << Iter::kWarpThreadC
    //    << " "
    //       << Iter::kWarpThreadS << std::endl;
    //    os << "[A] warp shape per access: " << Iter::kWarpAccessM << " "
    //       << " *** It represent the shape of warp in each global memory
    //       access.***" <<std::endl;
    //    os << "[A] warp access iters: " << Iter::kWarpIterM << " "
    //       << Iter::kWarpIterK << std::endl;
    //    os << "[A] warp arrangement: " << Iter::kWarpM << " " << Iter::kWarpK
    //       << std::endl;
    //    os << "[A] iterations: " << Iter::kIterM << " " << Iter::kIterK
    //       << " *** It represent the iterations in load global memory. *** "
    //       <<std::endl;
    //    os << "[A] iters per tile: " << Iter::kIterCount << std::endl;
    //    os << "[A] warp footprint: " << Iter::kWarpFootprintM << " "
    //       << Iter::kWarpFootprintK << std::endl;
    //    os << "[A] shared memory: " << Iter::kSmemByteSize << std::endl;
    //    os << std::endl;
    //  }
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
    // {

    //     using Iter = typename GemmType::IteratorQ;
    //     // os << "[Q] shape: " << CTA_M << " " << Iter::SLICE_K << std::endl;
    //     os << "[Q] warp thread arrangement: " << Iter::kWarpThreadC << " " <<
    //     Iter::kWarpThreadS << std::endl; os << "[Q] warp shape per access: "
    //     << Iter::kWarpAccessM << " " << Iter::kWarpAccessK << std::endl; os
    //     << "[Q] warp access iters: " << Iter::kWarpIterM << " " <<
    //     Iter::kWarpIterK << std::endl; os << "[Q] warp arrangement: " <<
    //     Iter::kWarpM << " " << Iter::kWarpK << std::endl; os << "[Q]
    //     iterations: " << Iter::kIterM << " " << Iter::kIterK << std::endl; os
    //     << "[Q] iters per tile: " << Iter::kIterCount << std::endl; os <<
    //     "[Q] warp footprint: " << Iter::kWarpFootprintM << " " <<
    //     Iter::kWarpFootprintK << std::endl; os << "[Q] size per stage: " <<
    //     Iter::kSizePerStage << std::endl; os << "[Q] shared memory: " <<
    //     Iter::kSmemByteSize << std::endl; os << std::endl;
    // }
    // os << "Dynamic shared memory size: " << kSmemByteSize << std::endl;
  }
};

template <typename TileShape, typename WarpShape, int Stages>
constexpr TileShape GemmKernel<TileShape, WarpShape, Stages>::tile_shape;

template <typename TileShape, typename WarpShape, int Stages>
constexpr WarpShape GemmKernel<TileShape, WarpShape, Stages>::warp_shape;

}  // namespace mmbenchmark
