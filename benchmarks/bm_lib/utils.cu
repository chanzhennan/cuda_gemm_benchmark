// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

#include "utils.h"

namespace cudabm {

std::string strFormatImp(const char* msg, va_list args) {
  // we might need a second shot at this, so pre-emptivly make a copy
  va_list args_cp;
  va_copy(args_cp, args);

  // TODO(ericwf): use std::array for first attempt to avoid one memory
  // allocation guess what the size might be
  std::array<char, 256> local_buff;

  // 2015-10-08: vsnprintf is used instead of snd::vsnprintf due to a limitation
  // in the android-ndk
  auto ret = vsnprintf(local_buff.data(), local_buff.size(), msg, args_cp);

  va_end(args_cp);

  // handle empty expansion
  if (ret == 0) return std::string{};
  if (static_cast<std::size_t>(ret) < local_buff.size())
    return std::string(local_buff.data());

  // we did not provide a long enough buffer on our first attempt.
  // add 1 to size to account for null-byte in size cast to prevent overflow
  std::size_t size = static_cast<std::size_t>(ret) + 1;
  auto buff_ptr = std::unique_ptr<char[]>(new char[size]);
  // 2015-10-08: vsnprintf is used instead of snd::vsnprintf due to a limitation
  // in the android-ndk
  vsnprintf(buff_ptr.get(), size, msg, args);
  return std::string(buff_ptr.get());
}

// adapted from benchmark srcs string utils
std::string strFormat(const char* format, ...) {
  va_list args;
  va_start(args, format);
  std::string tmp = strFormatImp(format, args);
  va_end(args);
  return tmp;
}

// template <typename T>
// void genRandom(std::vector<T>& vec) {
//   std::mt19937 gen;
//   std::uniform_real_distribution<> dist(-10.0, 10.0);
//   std::generate_n(vec.begin(), vec.size(), [&] { return dist(gen); });
// }

template <typename T>
void genRandom(T* vec, unsigned long len) {
  std::mt19937 gen;
  std::uniform_real_distribution<> dist(-10.0, 10.0);
  for (unsigned long i = 0; i < len; i++) {
    vec[i] = static_cast<T>(dist(gen));
  }
}

void genOnes(float* vec, unsigned long len) {
  for (unsigned long i = 0; i < len; i++) {
    vec[i] = 1.f;
  }
}

void Print(float* vec, size_t len) {
  for (int i = 0; i < len; i++) {
    printf("%f ", vec[i]);
    if (i % 10 == 0) {
      printf("\n");
    }
  }
}

float Sum(float* vec, size_t len) {
  float sum = 0.f;
  for (int i = 0; i < len; i++) {
    sum += vec[i];
  }
  return sum;
}

template <typename T>
void Gemm(T* dA, T* dB, T* dC, int m, int n, int k) {
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);

  // C = A X B
  if (std::is_same<T, float>::value) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                (float*)dB, n, (float*)dA, k, &beta, (float*)dC, n);

  } else if (std::is_same<T, __half>::value) {
    __half alpha = 1.0f;
    __half beta = 0.0f;
    cublasHgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                (__half*)dB, n, (__half*)dA, k, &beta, (__half*)dC, n);
  }

  cublasDestroy(blas_handle);
}

// Equal
template <typename Type>
bool Equal(const unsigned int n, const Type* x, const Type* y,
           const Type tolerance) {
  bool ok = true;

  for (unsigned int i = 0; i < n; ++i) {
    if (std::abs(x[i] - y[i]) > std::abs(tolerance)) {
      ok = false;
      return ok;
    }
  }

  return ok;
}

template bool Equal<float>(const unsigned int n, const float* x, const float* y,
                           const float tolerance);

template void genRandom<float>(float* vec, unsigned long len);
template void genRandom<half>(half* vec, unsigned long len);

template void Gemm<float>(float* dA, float* dB, float* dC, int m, int n, int k);
template void Gemm<__half>(__half* dA, __half* dB, __half* dC, int m, int n,
                           int k);

}  // namespace cudabm
