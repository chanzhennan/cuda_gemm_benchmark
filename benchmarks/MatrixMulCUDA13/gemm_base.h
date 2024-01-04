#pragma once
#include <cuda_fp16.h>

#include <iostream>
#include <memory>
#include <sstream>

struct iBaseGemm {
  virtual ~iBaseGemm() = default;

  // virtual void GetMetric(Metric& metric, int m, int n, int k) = 0;

  virtual void Launch(float* C, const float* A, const float* B, int M, int N,
                      int K) = 0;

  virtual void Dump(std::ostream& os) = 0;
};
