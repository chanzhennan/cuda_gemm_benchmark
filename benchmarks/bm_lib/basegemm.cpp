#include "basegemm.h"

template <typename T>
void BaseGemm<T>::callKernel(benchmark::State &state) {
  throw std::runtime_error("callKernel need implement");
}

template <typename T>
void BaseGemm<T>::SetUp(const ::benchmark::State &state) {
  // Populate array
  unsigned long M = (unsigned long)state.range(0);
  unsigned long N = (unsigned long)state.range(1);
  unsigned long K = (unsigned long)state.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;
  unsigned long csize = M * N;

  cudaMallocManaged(&dA, sizeof(T) * asize);
  cudaMallocManaged(&dB, sizeof(T) * bsize);
  cudaMallocManaged(&dC, sizeof(T) * csize);
  cudaMallocManaged(&testC, sizeof(T) * csize);

  genData(state);
}

template <typename T>
void BaseGemm<T>::genData(const ::benchmark::State &st) {
  unsigned long M = (unsigned long)st.range(0);
  unsigned long N = (unsigned long)st.range(1);
  unsigned long K = (unsigned long)st.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;

  cudabm::genRandom(dA, asize);
  cudabm::genRandom(dB, bsize);
}

template <typename T>
T *BaseGemm<T>::getDeviceA() {
  return dA;
}

template <typename T>
T *BaseGemm<T>::getDeviceB() {
  return dB;
}

template <typename T>
T *BaseGemm<T>::getDeviceC() {
  return dC;
}

template <typename T>
T *BaseGemm<T>::getDeviceTestC() {
  return testC;
}

template <typename T>
void BaseGemm<T>::verify(const ::benchmark::State &st) {
  // for test M, N, K = state.range(0)
  cudabm::Gemm<T>(dA, dB, testC, st.range(0), st.range(1), st.range(2));
  cudabm::Equal<T>(st.range(0) * st.range(1), dC, testC, 1e-2);
  // if (!)
  //   throw std::runtime_error("Value diff occur in Dense");
}

template <typename T>
void BaseGemm<T>::TearDown(const ::benchmark::State &st) {
  verify(st);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(testC);
}

template <typename T>
double BaseGemm<T>::getDataSize(const ::benchmark::State &state) {
  // datasize = 2 * M * N
  return (double)(2 * state.range(0) * state.range(1));
}

template <typename T>
double BaseGemm<T>::getFlops(const ::benchmark::State &state) {
  // flops =  2 * M * N * K / s
  return (double)(2 * long(state.range(0)) * state.range(1) * state.range(2));
}

template class BaseGemm<float>;
template class BaseGemm<__half>;
