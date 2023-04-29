// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);
void genRandom(float* vec, size_t len);
void Print(float* vec, size_t len);
float Sum(float* vec, size_t len);

void Gemm(float* dA, float* dB, float* dC, int m, int n, int k);

template <typename Type>
bool Equal(const unsigned int n, const Type* x, const Type* y,
           const Type tolerance);

}  // namespace cudabm
