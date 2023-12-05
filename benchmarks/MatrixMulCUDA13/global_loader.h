

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace mmbenchmark {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template <int WARPS, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES,
          int SLICES>
struct GlobalLoaderA {
  // blockK 按照slice分割，每次的大小就是slice_k
  static constexpr int SLICE_K = BLOCK_K / SLICES;  // 8
  static constexpr int kShapeK = 8;

  using AccessType = float;
  static constexpr int kAccessSize = sizeof(AccessType);

  static_assert(BLOCK_M % 32 == 0 && BLOCK_K % 8 == 0,
                "A is pre-formatted as 32x32 tiles");

  // A is [K/32, M/32, WARP_SIZE] uint4
  static constexpr int kShapeM = BLOCK_M;  // 128

  // thread access shape
  static constexpr int kAccessM = 4;
  static constexpr int kAccessK = 1;

  // warp thread arrangement
  static constexpr int kWarpThreadC = 4;
  static constexpr int kWarpThreadS = 8;

  // warp shape per access
  static constexpr int kWarpAccessM = kWarpThreadC * kAccessM;  // 16
  static constexpr int kWarpAccessK = kWarpThreadS * kAccessK;  // 8

  // warp access iterations
  static constexpr int kWarpIterM = kShapeM / kWarpAccessM;
  static constexpr int kWarpIterK = kShapeK / kWarpAccessK;

  // warp arrangement
  static constexpr int kWarpM = kWarpIterM >= WARPS ? WARPS : kWarpIterM;
  static constexpr int kWarpK = WARPS > kWarpIterM ? (WARPS / kWarpM) : 1;

  // iterations
  static constexpr int kIterM = kWarpIterM / kWarpM;
  static constexpr int kIterK = kWarpIterK / kWarpK;

  static constexpr int kIterCount = kIterM * kIterK;

  // warp footprint
  static constexpr int kWarpFootprintM = kWarpAccessM * kIterM;
  static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;

  static constexpr int kSizePerStage = kShapeK * kShapeM;
  static constexpr int kSmemByteSize = kAccessSize * STAGES * kSizePerStage;

  const uint* src_;
  void* smem_;
  int m_;
  int k_;
  int bm_;
  int bk_;
  int warp_id_;
  int lane_id_;

  __device__ GlobalLoaderA(const uint* src, void* smem, int m, int k, int bm,
                           int bk, int warp_id, int lane_id)
      : src_(src),
        smem_(smem),
        m_(m),
        k_(k),
        bm_(bm),
        bk_(bk),
        warp_id_(warp_id),
        lane_id_(lane_id) {
    const int warp_offset_m = warp_id_ % kWarpM;
    const int warp_offset_k = warp_id_ / kWarpM;

    const int warp_thread_offset_m = lane_id_ % kWarpThreadC;
    const int warp_thread_offset_k = lane_id_ / kWarpThreadC;

    const int block_thread_offset_m =
        kWarpFootprintM * warp_offset_m + warp_thread_offset_m * kAccessM;
    const int block_thread_offset_k =
        kWarpFootprintK * warp_offset_k + warp_thread_offset_k * kAccessK;

    const int src_offset_m = block_thread_offset_m + bm_;
    const int src_offset_k = block_thread_offset_k + bk_ / 32;
  }
};

template <int WARPS, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES,
          int SLICES>
struct GlobalLoaderB {
  const uint* src_;
  void* smem_;

  int n_;
  int k_;
  int bn_;
  int bk_;
  int warp_id_;
  int lane_id_;

  __device__ GlobalLoaderB(const uint* src, void* smem, int k, int n, int bk,
                           int bn, int warp_id, int lane_id)
      : src_(src),
        smem_(smem),
        k_(k),
        n_(n),
        bk_(bk),
        bn_(bn),
        warp_id_(warp_id),
        lane_id_(lane_id) {}
};

}  // namespace mmbenchmark
