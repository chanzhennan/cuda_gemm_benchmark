#include "MatrixMulCUDA9/cublas.cuh"

template <typename T>
void GEMM9(T *dA, T *dB, T *dC, int m, int n, int k) {
  // C = A X B
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  if (std::is_same<T, float>::value) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                (float *)dB, m, (float *)dA, k, &beta, (float *)dC, m);

  } else if (std::is_same<T, __half>::value) {
    __half alpha = 1.0f;
    __half beta = 0.0f;
    cublasHgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                (__half *)dB, m, (__half *)dA, k, &beta, (__half *)dC, m);
  }
  cublasDestroy(blas_handle);
}

template void GEMM9<float>(float *dA, float *dB, float *dC, int m, int n,
                           int k);
template void GEMM9<__half>(__half *dA, __half *dB, __half *dC, int m, int n,
                            int k);
