// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#define checkCuda(call)                                                 \
  do {                                                                  \
    cudaError_t status = (call);                                        \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " (" \
                << status << ") at " << __FILE__ << ":" << __LINE__     \
                << std::endl;                                           \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define checkCuBlasErrors(func)                                               \
  {                                                                           \
    cublasStatus_t e = (func);                                                \
    if (e != CUBLAS_STATUS_SUCCESS)                                           \
      printf("%s %d CuBlas: %s", __FILE__, __LINE__, _cuBlasGetErrorEnum(e)); \
  }

namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);
void genRandom(float* vec, unsigned long len);
void genOnes(float* vec, unsigned long len);
void Print(float* vec, size_t len);
float Sum(float* vec, size_t len);

void Gemm(float* dA, float* dB, float* dC, int m, int n, int k);

template <typename Type>
bool Equal(const unsigned int n, const Type* x, const Type* y,
           const Type tolerance);

}  // namespace cudabm
