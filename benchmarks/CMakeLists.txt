add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  MatrixMulCUDA0/naive.cc
  MatrixMulCUDA0/naive.cu
  # MatrixMulCUDA1/blocker.cc MatrixMulCUDA1/blocker.cu
  # MatrixMulCUDA2/strider.cc MatrixMulCUDA2/strider.cu
  # MatrixMulCUDA3/aligner.cc MatrixMulCUDA3/aligner.cu
  # MatrixMulCUDA4/multiloader.cc MatrixMulCUDA4/multiloader.cu
  # MatrixMulCUDA5/bankconflictavoider.cc MatrixMulCUDA5/bankconflictavoider.cu
  # MatrixMulCUDA6/pingpongbuffer.cc MatrixMulCUDA6/pingpongbuffer.cu #
  # MatrixMulCUDA7/sgemm128.cc MatrixMulCUDA7/sgemm128.cu
  # MatrixMulCUDA8/dense.cc MatrixMulCUDA8/dense.cu MatrixMulCUDA9/cublas.cc
  # MatrixMulCUDA9/cublas.cu MatrixMulCUDA10/yzaiustc.cc
  # MatrixMulCUDA10/yzaiustc.cu MatrixMulCUDA11/yhs.cc MatrixMulCUDA11/yhs.cu
  # MatrixMulCUDA12/czn.cc MatrixMulCUDA12/czn.cu MatrixMulCUDA13/czn_k16.cc
  # MatrixMulCUDA13/czn_k16.cu
)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
cuda_add_cublas_to_target(cuda_benchmark)

set_property(TARGET cuda_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(cuda_benchmark bm_lib)

install(TARGETS cuda_benchmark RUNTIME)
