add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  baseline/baseline.cc
  baseline/baseline.cu
  # no_divergence_branch/no_divergence_branch.cc
  # no_divergence_branch/no_divergence_branch.cu
  # no_bank_conflict/no_bank_conflict.cc
  # no_bank_conflict/no_bank_conflict.cu
  # add_during_load/add_during_load.cc
  # add_during_load/add_during_load.cu
  # unroll_last_warp/unroll_last_warp.cc
  # unroll_last_warp/unroll_last_warp.cu
  # completely_unroll/completely_unroll.cc
  # completely_unroll/completely_unroll.cu
  # shuffle/shuffle.cc
  # shuffle/shuffle.cu
  )

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
target_link_libraries(cuda_benchmark bm_lib)

install(TARGETS cuda_benchmark RUNTIME)
