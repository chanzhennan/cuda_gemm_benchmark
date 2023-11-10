#include "basegemm.h"

void BaseGemm::callKernel(benchmark::State &state) {
  throw std::runtime_error("callKernel need implement");
}

void BaseGemm::SetUp(const ::benchmark::State &state) {
  // Populate array
  unsigned long M = (unsigned long)state.range(0);
  unsigned long N = (unsigned long)state.range(1);
  unsigned long K = (unsigned long)state.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;
  unsigned long csize = M * N;

  cudaMallocManaged(&dA, sizeof(float) * asize);
  cudaMallocManaged(&dB, sizeof(float) * bsize);
  cudaMallocManaged(&dC, sizeof(float) * csize);
  cudaMallocManaged(&testC, sizeof(float) * csize);

  genData(state);
}

void BaseGemm::genData(const ::benchmark::State &st) {
  unsigned long M = (unsigned long)st.range(0);
  unsigned long N = (unsigned long)st.range(1);
  unsigned long K = (unsigned long)st.range(2);
  unsigned long asize = M * K;
  unsigned long bsize = K * N;

  cudabm::genRandom(dA, asize);
  cudabm::genRandom(dB, bsize);
}

float *BaseGemm::getDeviceA() { return dA; }

float *BaseGemm::getDeviceB() { return dB; }

float *BaseGemm::getDeviceC() { return dC; }

float *BaseGemm::getDeviceTestC() { return testC; }

void BaseGemm::verify(const ::benchmark::State &st) {
  // for test M, N, K = state.range(0)
  cudabm::Gemm(dA, dB, testC, st.range(0), st.range(1), st.range(2));
  if (!cudabm::Equal<float>(st.range(0) * st.range(1), dC, testC, 1e-2))
    throw std::runtime_error("Value diff occur in Dense");
}

void BaseGemm::TearDown(const ::benchmark::State &st) {
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(testC);
}

double BaseGemm::getDataSize(const ::benchmark::State &state) {
  // datasize = 2 * M * N
  return (double)(2 * state.range(0) * state.range(1));
}

double BaseGemm::getFlops(const ::benchmark::State &state) {
  // flops =  2 * M * N * K / s
  return (double)(2 * long(state.range(0)) * state.range(1) * state.range(2));
}
