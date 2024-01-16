
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "utils.h"
template <typename T>
class BaseGemm : public benchmark::Fixture {
 public:
  virtual void callKernel(benchmark::State &state);

  void SetUp(const ::benchmark::State &state) override;

  void genData(const ::benchmark::State &st);

  void verify(const ::benchmark::State &st);

  void TearDown(const ::benchmark::State &st) override;

  double getDataSize(const ::benchmark::State &state);

  double getFlops(const ::benchmark::State &state);

  T *getDeviceA();
  T *getDeviceB();
  T *getDeviceC();
  T *getDeviceTestC();

 private:
  T *dA, *dB;
  T *testC, *dC;
  long int dataSize;
  long int flops;
};
