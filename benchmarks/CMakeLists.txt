add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  MatrixMulCUDA0/naive.cc
  MatrixMulCUDA0/naive.cu
  MatrixMulCUDA1/blocker.cc
  MatrixMulCUDA1/blocker.cu
  MatrixMulCUDA2/strider.cc
  MatrixMulCUDA2/strider.cu
  MatrixMulCUDA3/aligner.cc
  MatrixMulCUDA3/aligner.cu
  # outer_memory/outer_memory.cc outer_memory/outer_memory.cu
  # completely_unroll/completely_unroll.cc
  # completely_unroll/completely_unroll.cu define_extend/define_extend.cc
  # define_extend/define_extend.cu interleave/interleave.cc
  # interleave/interleave.cu
)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
set_property(TARGET cuda_benchmark PROPERTY CUDA_ARCHITECTURES 89)
target_link_libraries(cuda_benchmark bm_lib cublas)

install(TARGETS cuda_benchmark RUNTIME)
