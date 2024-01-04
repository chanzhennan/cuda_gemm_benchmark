#include <cstddef>
#include <cstdint>

#include "common.h"

namespace mmbenchmark {

template<int BLOCK_M,
         int BLOCK_K,
         int WARP_M,
         int WARP_K,
         int OP_M,
         int OP_K,
         int STAGES,
         int kSizePerStageA,
         int kSizePerStageQ>
struct WarpIteratorA {

    static_assert(WARP_K % GROUP_SIZE == 0 || GROUP_SIZE % WARP_K == 0);

    // static constexpr int ITER_M = 32 / OP_M;
    // static constexpr int ITER_X = WARP_M / 32;

    // uint4 frag_A4_[ITER_X];    // 8 value per uint
    // half2 frag_Q_[ITER_X][4];  // 4 m8k8 tile along M, as WARP_M == 32

    // const uint4* smem_A_;
    // const half2* smem_Q_;
    // const int    offset_m_;
    // const int    offset_m_Q_;

    // int stage_{0};
    // int offset_A_{0};
    // int offset_Q_{0};

    __device__ WarpIteratorA(uint4* smem_A, half2* smem_Q, int warp_id, int lane_id, int offset_m, int offset_k):
        smem_A_(smem_A), smem_Q_(smem_Q), offset_m_(offset_m), offset_m_Q_(offset_m / 32 * 32 + lane_id / 4) {}

    iter_k must be a compile tile constant
    __device__ void load(Array<half, 8>* data, int iter_k)
    {
        load A
        smem_A uint4 [SLICE_K/32, CTA_M/32, WARP_SIZE], load as uint4 to avoid bank-conflicts
        if (iter_k % 2 == 0) {
            PRAGMA_UNROLL
            for (int x = 0; x < ITER_X; ++x) {
                frag_A4_[x] = smem_A_[offset_A_ + (iter_k / 2) * CTA_M + x * 32 + offset_m_];
            }
        }

        load Q
        if (iter_k * OP_K % GROUP_SIZE == 0) {
            const int g = iter_k * OP_K / GROUP_SIZE;
            PRAGMA_UNROLL
            for (int x = 0; x < ITER_X; ++x) {
                PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const int mm           = offset_m_Q_ + x * 32 + i * 8;  // stride of m8k8 tile
                    ((uint&)frag_Q_[x][i]) = ((uint&)smem_Q_[offset_Q_ + g * CTA_M + mm]);
                }
            }
        }

        PRAGMA_UNROLL
        for (int x = 0; x < ITER_X; ++x) {
            const uint* frag_A = (uint*)&frag_A4_[x];
            PRAGMA_UNROLL
            for (int iter_m = 0; iter_m < ITER_M; ++iter_m) {
                uint4 tmp = dequantize_s4_to_fp16x2_v2(frag_A[iter_k % 2 * 2 + iter_m]);
                auto& vec = (Array<half2, 4>&)tmp;

                vec[0] = apply_Q(vec[0], frag_Q_[x][iter_m * 2]);
                vec[1] = apply_Q(vec[1], frag_Q_[x][iter_m * 2 + 1]);
                vec[2] = apply_Q(vec[2], frag_Q_[x][iter_m * 2]);
                vec[3] = apply_Q(vec[3], frag_Q_[x][iter_m * 2 + 1]);

                data[x * ITER_M + iter_m] = (Array<half, 8>&)vec;
            }
        }
    }

    // __device__ void next_stage()
    // {
    //     ++stage_;
    //     if (stage_ >= STAGES) {
    //         stage_ = 0;
    //     }
    //     offset_A_ = stage_ * kSizePerStageA;
    //     offset_Q_ = stage_ * kSizePerStageQ;
    // }
};

}