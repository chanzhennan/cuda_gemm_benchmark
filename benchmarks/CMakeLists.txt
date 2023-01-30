add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  baseline/baseline.cc
  baseline/baseline.cu
  unroll/unroll.cc
  unroll/unroll.cu
  stride/stride.cc
  stride/stride.cu
  fix_block_size/fix_block_size.cc
  fix_block_size/fix_block_size.cu
  outer_memory/outer_memory.cc
  outer_memory/outer_memory.cu
  # completely_unroll/completely_unroll.cc
  # completely_unroll/completely_unroll.cu shuffle/shuffle.cc shuffle/shuffle.cu
)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
target_link_libraries(cuda_benchmark bm_lib)

install(TARGETS cuda_benchmark RUNTIME)
