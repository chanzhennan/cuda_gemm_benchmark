#include "MatrixMulCUDA9/cublas.cuh"

// refer to MatrixMulCUDA8 @
// https://github.com/Cjkkkk/CUDA_gemm/blob/master/src/cuda/quantization_8bit.cu
template <typename T>
void GEMM9(T *dA, T *dB, T *dC, int m, int n, int k,
           cublasHandle_t &blas_handle) {
  float alpha = 1.0f;
  float beta = 0.0f;

  // C = A X B
  cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, n, dA,
              k, &beta, dC, n);
  cudaDeviceSynchronize();
}

template void GEMM9<float>(float *dA, float *dB, float *dC, int m, int n, int k,
                           cublasHandle_t &blas_handle);
