

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace mmbenchmark {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template <typename T>
__inline__ __device__ void cp_async_cg_A(uint32_t smem_int_ptr,
                                         const T* __restrict__ src, bool mask) {
#if TURBOMIND_ARCH_SM80
  constexpr int cp_size = sizeof(T);
  static_assert(cp_size == 16, "cp.async.cg requreis cp_size == 16");
  // clang-format off
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global" L2_CACHEHINT(256) " [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr),
                 "l"(src),
                 "n"(cp_size));
  // clang-format on
#else
  assert(TURBOMIND_ARCH_SM80);
#endif
}

template <typename T>
__inline__ __device__ void cp_async_cg_B(uint32_t smem_int_ptr,
                                         const T* __restrict__ src, bool mask) {
#if TURBOMIND_ARCH_SM80
  constexpr int cp_size = sizeof(T);
  static_assert(cp_size == 16, "cp.async.cg requreis cp_size == 16");
  // clang-format off
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  setp.ne.b32 p, %0, 0;\n"
                 "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                 "}\n" ::"r"((int)mask),
                 "r"(smem_int_ptr),
                 "l"(src),
                 "n"(cp_size));
  // clang-format on
#else
  assert(TURBOMIND_ARCH_SM80);
#endif
}

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
  static constexpr int kWarpAccessM = kWarpThreadC * kAccessM;
  static constexpr int kWarpAccessK =
      kWarpThreadS * kAccessK;  // 一个warp k维度能处理的个数   1 * 1 = 1

  // warp access iterations
  static constexpr int kWarpIterM = kShapeM / kWarpAccessM;  // 128 / 128 = 1
  static constexpr int kWarpIterK = kShapeK / kWarpAccessK;  // 8 / 1 = 8

  // warp arrangement
  static constexpr int kWarpM = kWarpIterM >= WARPS ? WARPS : kWarpIterM;   // 8
  static constexpr int kWarpK = WARPS > kWarpIterM ? (WARPS / kWarpM) : 1;  // 1

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
  float* smem_;
  uint32_t smem_int_ptr_;

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

  int iter_m_{0};

  __device__ GlobalLoaderA(const float* src, float* smem, int m, int k, int bm,
                           int bk, int warp_id, int lane_id)
      : src_(src),
        smem_(smem),
        smem_int_ptr_(cast_smem_ptr_to_uint(smem)),
        m_(m),
        k_(k),
        bm_(bm),
        bk_(bk),
        warp_id_(warp_id),
        lane_id_(lane_id) {
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //   printf("GlobalLoaderA kWarpM = %d kWarpM = %d kIterM = %d kIterK = %d
    //   kIterCount = %d \ kWarpIterM = %d kWarpIterK = %d kIterM = %d kIterK =
    //   %d kSmemByteSize = %d\n", kWarpM, kWarpK, kIterM, kIterK, kIterCount,
    //   kWarpIterM, kWarpIterK, kIterM, kIterK, kSmemByteSize);
    // }

    const int warp_offset_m = warp_id_ % kWarpM;
    const int warp_offset_k = warp_id_ / kWarpM;

    const int warp_thread_offset_m = lane_id_ % kWarpThreadC;
    const int warp_thread_offset_k = lane_id_ / kWarpThreadC;

    const int cta_thread_offset_m =
        kWarpFootprintM * warp_offset_m + warp_thread_offset_m * kAccessM;
    const int cta_thread_offset_k =
        kWarpFootprintK * warp_offset_k + warp_thread_offset_k * kAccessK;

    const int src_offset_m = cta_thread_offset_m + bm_;
    const int src_offset_k = cta_thread_offset_k + bk_ / 32;

    // if (threadIdx.x < 64 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //   printf("tid = %d, warp_offset_m = %d warp_offset_k = %d \
        //   warp_thread_offset_m = %d warp_thread_offset_k = %d    \
        //   cta_thread_offset_m = %d cta_thread_offset_k = %d \
        //   src_offset_m = %d src_offset_k = %d \n", threadIdx.x, warp_offset_m,
    //   warp_offset_k, warp_thread_offset_m, warp_thread_offset_k,
    //   cta_thread_offset_m, cta_thread_offset_k, src_offset_m, src_offset_k);
    // }

    src_offset_ = src_offset_k * m_ + src_offset_m;
    src_step_m_ = kWarpAccessM;
    src_step_k_ = kWarpAccessK * m_ - kIterM * kWarpAccessM;
    src_step_s_ = BLOCK_M / 32 * m_ - kIterK * kWarpAccessK * m_;

    const int dst_offset_m = cta_thread_offset_m;
    const int dst_offset_k = cta_thread_offset_k;

    dst_offset_ = dst_offset_k * kShapeM + dst_offset_m;
    dst_step_m_ = kWarpAccessM;
    dst_step_k_ = kWarpAccessK * kShapeM - kIterM * kWarpAccessM;
    dst_step_s_ = SLICE_K / 32 * kShapeM - kIterK * kWarpAccessK * kShapeM;

    dst_offset_ *= kAccessSize;
    dst_step_m_ *= kAccessSize;
    dst_step_k_ *= kAccessSize;
    dst_step_s_ *= kAccessSize;

    // if (threadIdx.x ==0 && blockIdx.x == 1 && blockIdx.y == 1) {
    //   printf("tid = %d, bm = %d bk = %d\n", threadIdx.x, bm_, bk_);
    // }

    // if (threadIdx.x < 32 && blockIdx.x == 1 && blockIdx.y == 1){
    //   printf("tid = %d src_offset_m = %d dst_offset_m = %d \n", threadIdx.x,
    //   src_offset_m, dst_offset_m); printf("tid = %d src_offset_k = %d
    //   dst_offset_k = %d \n", threadIdx.x, src_offset_k, dst_offset_k);
    //   // printf("tid = %d, src_offset_m = %d src_offset_k = %d\n",
    //   threadIdx.x, src_offset_m, src_offset_k);
    //   // printf("tid = %d, cta_thread_offset_m = %d cta_thread_offset_k =
    //   %d\n", threadIdx.x, cta_thread_offset_m, cta_thread_offset_k);
    // }
  }

  __device__ void prefetch_stage(bool mask) {
    for (int i = 0; i < kIterCount; ++i) {
      prefetch(mask);
      ++(*this);
    }
    // next_stage();
  }

  __device__ GlobalLoaderA& operator++() {
    src_offset_ += src_step_m_;
    dst_offset_ += src_step_k_;
    ++iter_m_;
    if (iter_m_ < kIterM) {
      return *this;
    }
    iter_m_ = 0;
    src_offset_ += src_step_k_;
    dst_offset_ += dst_step_k_;

    return *this;
  }

  __device__ void prefetch(bool mask) {
    if (threadIdx.x < 10 && blockIdx.x == 0 && blockIdx.y == 0) {
      printf(
          "xxxx tid = %d, src_offset_ = %d dst_offset_ = %d smem_int_ptr_= "
          "%d\n",
          threadIdx.x, src_offset_, dst_offset_, smem_int_ptr_);
    }

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
  static constexpr int SLICE_K = BLOCK_K / SLICES;  // 8
  static constexpr int kElementSize = sizeof(float);
  using AccessType = float4;
  static constexpr int kAccessSize = sizeof(AccessType);

  static constexpr int kShapeK = SLICE_K;
  static constexpr int kShapeN = BLOCK_N;

  static constexpr int kAccessN = kAccessSize / sizeof(float);

  // static_assert(kShapeK % kAccessK == 0);

  // warp thread arrangement
  static constexpr int kWarpThreadC = std::max(kShapeN / kAccessN, 1);
  static constexpr int kWarpThreadS = WARP_SIZE / kWarpThreadC;

  // warp shape per access
  static constexpr int kWarpAccessN = kWarpThreadC * kAccessN;
  static constexpr int kWarpAccessK = kWarpThreadS;

  // warp access iterations
  static constexpr int kWarpIterN = kShapeN / kWarpAccessN;
  static constexpr int kWarpIterK = kShapeK / kWarpAccessK;

  // warp arrangement
  static constexpr int kWarpN = kWarpIterN >= WARPS ? WARPS : kWarpIterN;
  static constexpr int kWarpK = WARPS > kWarpIterN ? WARPS / kWarpN : 1;

  // iterations
  static constexpr int kIterK = kWarpIterK / kWarpK;
  static constexpr int kIterN = kWarpIterN >= kWarpN ? kWarpIterN / kWarpN : 1;

  static constexpr int kIterCount = kIterK * kIterN;
  static_assert(kIterCount > 0);

  // warp footprint
  static constexpr int kWarpFootprintK = kWarpAccessK * kIterK;
  static constexpr int kWarpFootprintN = kWarpAccessN * kIterN;

  // Eliminate bank-conflicts for float4 aligned
  static constexpr int kSmemPadCtaN = BLOCK_N + 4;
  // todo 暂定认为不需要迭代
  static constexpr int kSizePerTile = SLICE_K * kSmemPadCtaN;
  static constexpr int kSmemByteSize = kElementSize * STAGES * kSizePerTile;

  const float* src_;
  void* smem_;

  int n_;
  int k_;
  int bn_;
  int bk_;
  int warp_id_;
  int lane_id_;

  int src_offset_n_;

  int src_offset_;
  int dst_offset_;

  int src_step_k_;
  int src_step_n_;
  int dst_step_k_;
  int dst_step_n_;
  bool is_valid_n_;

  int tmp_src_offset_;
  int tmp_dst_offset_;
  int tmp_src_offset_n_;

  uint32_t smem_int_ptr_;

  __device__ GlobalLoaderB(const float* src, void* smem, int k, int n, int bk,
                           int bn, int warp_id, int lane_id)
      : src_(src),
        smem_((AccessType*)smem),
        smem_int_ptr_(cast_smem_ptr_to_uint(smem)),
        k_(k),
        n_(n),
        bk_(bk),
        bn_(bn),
        warp_id_(warp_id),
        lane_id_(lane_id) {
    const int warp_offset_k = warp_id_ % kWarpK;
    const int warp_offset_n = warp_id_ / kWarpK;

    const int warp_thread_offset_n = lane_id_ % kWarpThreadC;
    const int warp_thread_offset_k = lane_id_ / kWarpThreadC;

    const int tile_thread_offset_k =
        kWarpFootprintK * warp_offset_k + warp_thread_offset_k;
    const int tile_thread_offset_n =
        kWarpFootprintN * warp_offset_n + warp_thread_offset_n * kAccessN;

    const int src_offset_k = tile_thread_offset_k + bk_;
    src_offset_n_ = tile_thread_offset_n + bn_;

    const int dst_offset_k = tile_thread_offset_k;
    const int dst_offset_n = tile_thread_offset_n;

    src_offset_ = src_offset_k * n_ + src_offset_n_;
    dst_offset_ = dst_offset_k * kSmemPadCtaN + dst_offset_n;

    // Update, because *src type is float, *smem type is void*

    src_step_k_ = kWarpAccessK;
    // src_step_n_ = kWarpAccessN * k_ - kIterK * kWarpAccessK;

    dst_step_k_ = kWarpAccessK;
    // dst_step_n_ = kWarpAccessN * SLICE_K - kIterK * kWarpAccessK;

    dst_offset_ *= kElementSize;
    dst_step_k_ *= kElementSize;
    dst_step_n_ *= kElementSize;

    // if (threadIdx.x < 64 && blockIdx.x == 0 && blockIdx.y == 0){
    //   // printf("tid = %d tile_thread_offset_k = %d tile_thread_offset_n = %d
    //   \n", threadIdx.x, tile_thread_offset_k, tile_thread_offset_n);
    //   // printf("tid = %d tile_thread_offset_k = %d tile_thread_offset_n = %d
    //   \n", threadIdx.x, tile_thread_offset_k, tile_thread_offset_n);
    //   printf("tid = %d, src_offset_ = %d dst_offset_ = %d\n", threadIdx.x,
    //   src_offset_, dst_offset_);
    //   // printf("tid = %d, cta_thread_offset_m = %d cta_thread_offset_k =
    //   %d\n", threadIdx.x, cta_thread_offset_m, cta_thread_offset_k);
    // }

    tmp_src_offset_ = src_offset_;
    tmp_dst_offset_ = dst_offset_;

    tmp_src_offset_n_ = src_offset_n_;
    is_valid_n_ = tmp_src_offset_n_ < n_;
  }

  __device__ void prefetch_stage(bool mask) {
    PRAGMA_UNROLL
    for (int i = 0; i < kIterCount; ++i) {
      prefetch(mask);
      // ++(*this);
    }
    // next_stage();
  }

  __device__ void prefetch(bool mask) {
    if (threadIdx.x < 32 && blockIdx.x == 0 && blockIdx.y == 0) {
      // printf("tid = %d tile_thread_offset_k = %d tile_thread_offset_n = %d
      // \n", threadIdx.x, tile_thread_offset_k, tile_thread_offset_n);
      // printf("tid = %d tile_thread_offset_k = %d tile_thread_offset_n = %d
      // \n", threadIdx.x, tile_thread_offset_k, tile_thread_offset_n);
      printf("tid = %d, tmp_src_offset_ = %d tmp_dst_offset_ = %d\n",
             threadIdx.x, tmp_src_offset_, tmp_dst_offset_);
      // printf("tid = %d, cta_thread_offset_m = %d cta_thread_offset_k = %d\n",
      // threadIdx.x, cta_thread_offset_m, cta_thread_offset_k);
    }
#if TURBOMIND_ARCH_SM80
    cp_async_cg_B(smem_int_ptr_ + tmp_dst_offset_,
                  (const AccessType*)(src_ + tmp_src_offset_),
                  is_valid_n_ && mask);
#else
    if (is_valid_n_ && mask) {
      *(AccessType*)((uint8_t*)smem_ + tmp_dst_offset_) =
          __ldg((const AccessType*)(src_ + tmp_src_offset_));
    }
#endif
  }
};

}  // namespace mmbenchmark
