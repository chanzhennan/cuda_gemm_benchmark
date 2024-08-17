#include <cstddef>
#include <cstdint>

#include "common.h"

namespace mmbenchmark {

__device__ __forceinline__ void lds128(uint32_t &reg0, uint32_t &reg1,
                                       uint32_t &reg2, uint32_t &reg3,
                                       const uint32_t &addr) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
               : "r"(addr));
}

template <int TILE_M, int TILE_K, int WARP_M, int WARP_K, int kSizePerStageA,
          int STAGES>
struct WarpIteratorA {
  // static_assert(WARP_K % GROUP_SIZE == 0 || GROUP_SIZE % WARP_K == 0);

  uint32_t smem_base_ptr_;

  int stride_m_ = 0;
  int warp_id_m_;
  int offset_k_;

  int offset_m_;

  int stage_{0};

  __device__ WarpIteratorA(uint32_t smem_int_ptr, int warp_id_m, int offset_m,
                           int offset_k)
      : smem_base_ptr_(smem_int_ptr),
        offset_m_(offset_m),
        warp_id_m_(warp_id_m),
        offset_k_(offset_k) {
    int i = 0;
  }

  // iter_k must be a compile tile constant
  __device__ void load(Array<half, 8> &data, int iter_k) {
    // load A
    auto ptr = (uint32_t *)data.data();
    auto src = smem_base_ptr_ + sizeof(half) * iter_k * stride_m_;
    lds128(ptr[0], ptr[2], ptr[4], ptr[8], src);
  }

  __device__ void next_stage() {
    ++stage_;
    if (stage_ >= STAGES) {
      stage_ = 0;
    }
    // offset_A_ = stage_ * kSizePerStageA;
  }
};

template <int TILE_N, int TILE_K, int WARP_N, int WARP_K, int SMEM_STRIDN,
          int STAGES>
struct WarpIteratorB {
  static constexpr int ITER_N = WARP_N / OP_N;
  static constexpr int ITER_K = WARP_K / OP_K;

  int warp_id_n_;
  int stride_n_;
  int offset_n_;
  int offset_k_;

  const uint32_t smem_base_ptr_;

  int stage_{0};

  __device__ WarpIteratorB(uint32_t smem_int_ptr, int warp_id_n, int offset_n,
                           int offset_k)
      : smem_base_ptr_(smem_int_ptr),
        warp_id_n_(warp_id_n),
        offset_n_(offset_n),
        offset_k_(offset_k) {}

  __device__ void load(Array<half, 8> &data, int iter_k) {
    auto ptr = (uint32_t *)data.data();
    auto src = smem_base_ptr_ + sizeof(half) * iter_k * stride_n_;
    lds128(ptr[0], ptr[2], ptr[4], ptr[6], src);
  }

  __device__ void next_stage() {
    ++stage_;
    if (stage_ >= STAGES) {
      stage_ = 0;
    }
    // smem_ptr_ = smem_base_ptr_ + stage_ * sizeof(half) * TILE_N *
    // SMEM_STRIDN;
  }
};

}  // namespace mmbenchmark
