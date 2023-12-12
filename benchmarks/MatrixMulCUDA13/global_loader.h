

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
  // 因为这个struct主要是block thread拷贝 所以可以不考虑warp的排布，直接32*1
  // 拉开，有多少global数据宝贝，就直接拷贝多少个数
  static constexpr int SLICE_K = BLOCK_K / SLICES;  // 8 / 1

  using AccessType = float;  // 一个线程拷贝float4
  static constexpr int kAccessSize = sizeof(AccessType);

  static_assert(BLOCK_M % 32 == 0 && BLOCK_K % 1 == 0,
                "A is pre-formatted as 32x32 tiles");

  // A is [K/32, M/32, WARP_SIZE] uint4
  static constexpr int kShapeM = BLOCK_M;  // 128
  static constexpr int kShapeK = SLICE_K;  // 8

  // thread access shape
  static constexpr int kAccessM = 4;  //一个线程处理一个float4
  static constexpr int kAccessK = 1;  //一个线程数里一个float

  // warp thread arrangement
  static constexpr int kWarpThreadC = 32;  // 直接平铺，
  static constexpr int kWarpThreadS = 1;

  // warp shape per access
  static constexpr int kWarpAccessM =
      kWarpThreadC * kAccessM;  // 一个warp m维度能处理的个数   32 * 4 = 128
  static constexpr int kWarpAccessK =
      kWarpThreadS * kAccessK;  // 一个warp k维度能处理的个数   1 * 1 = 1

  // warp access iterations
  static constexpr int kWarpIterM = kShapeM / kWarpAccessM;  // 128 / 128 = 1
  static constexpr int kWarpIterK = kShapeK / kWarpAccessK;  // 8 / 1 = 8

  // warp arrangement
  static constexpr int kWarpM = kWarpIterM >= WARPS ? WARPS : kWarpIterM;   // 1
  static constexpr int kWarpK = WARPS > kWarpIterM ? (WARPS / kWarpM) : 1;  // 8

  // iterations
  static constexpr int kIterM = kWarpIterM / kWarpM;  // 1
  static constexpr int kIterK = kWarpIterK / kWarpK;  // 1

  static constexpr int kIterCount = kIterM * kIterK;  // 1

  // warp footprint
  static constexpr int kWarpFootprintM = kWarpAccessM * kIterM;  // 128 * 1
  static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;  // 1 * 1

  static constexpr int kSizePerStage = kShapeK * kShapeM;  // 128 * 8
  static constexpr int kSmemByteSize =
      kAccessSize * STAGES * kSizePerStage;  // 128 * 8 * 3

  const float* src_;
  void* smem_;
  int m_;
  int k_;
  int bm_;
  int bk_;
  int warp_id_;
  int lane_id_;

  int src_offset_;
  int dst_offset_;

  int src_step_m_;
  int src_step_k_;
  int src_step_s_;

  int dst_step_m_;
  int dst_step_k_;
  int dst_step_s_;

  __device__ GlobalLoaderA(const float* src, void* smem, int m, int k, int bm,
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

  __device__ void prefetch_stage(bool mask) {
    for (int i = 0; i < kIterCount; ++i) {
      prefetch(mask);
      ++(*this);
    }
    // next_stage();
  }

  __device__ void prefetch(bool mask) {
#if TURBOMIND_ARCH_SM80
    cp_async_cg_A(smem_int_ptr_ + dst_offset_,
                  (const AccessType*)src_ + src_offset_, mask);
#else
    if (mask) {
      *(AccessType*)((uint8_t*)smem_ + dst_offset_) =
          __ldg((const AccessType*)src_ + src_offset_);
    }
#endif
  }
};

template <int WARPS, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES,
          int SLICES>
struct GlobalLoaderB {
  static constexpr int SLICE_K = BLOCK_K / SLICES;
  static constexpr int kElementSize = sizeof(float);
  using AccessType = float;
  static constexpr int kAccessSize = sizeof(AccessType);

  static constexpr int kShapeK = SLICE_K;
  static constexpr int kShapeN = BLOCK_N;

  static constexpr int kAccessK = kAccessSize / sizeof(float);

  //   static_assert(kShapeK % kAccessSize == 0);

  // warp thread arrangement
  static constexpr int kWarpThreadC = std::max(kShapeK / kAccessK, 1);
  static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

  // warp shape per access
  static constexpr int kWarpAccessK = kWarpThreadC * kAccessK;
  static constexpr int kWarpAccessN = kWarpThreadS;

  // warp access iterations
  static constexpr int kWarpIterK = kShapeK / kWarpAccessK;
  static constexpr int kWarpIterN = kShapeN / kWarpAccessN;

  // warp arrangement
  static constexpr int kWarpK = kWarpIterK >= WARPS ? WARPS : kWarpIterK;
  static constexpr int kWarpN = WARPS > kWarpIterK ? WARPS / kWarpK : 1;

  // iterations
  static constexpr int kIterK = kWarpIterK / kWarpK;
  static constexpr int kIterN = kWarpIterN >= kWarpN ? kWarpIterN / kWarpN : 1;

  static constexpr int kIterCount = kIterK * kIterN;
  static_assert(kIterCount > 0);

  // warp footprint
  static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;
  static constexpr int kWarpFootprintN = kWarpAccessN * kIterN;

  // Eliminate bank-conflicts for 8x4 half2 tiles, watch out for misalignment
  static constexpr int kSmemPadCtaK = SLICE_K + 8;
  static constexpr int kSizePerTile = BLOCK_N * kSmemPadCtaK;
  static constexpr int kSmemByteSize = kElementSize * STAGES * kSizePerTile;

  const float* src_;
  void* smem_;

  int n_;
  int k_;
  int bn_;
  int bk_;
  int warp_id_;
  int lane_id_;

  __device__ GlobalLoaderB(const float* src, void* smem, int k, int n, int bk,
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
