// Copyright (c) 2022 Zhennanc Ltd. All rights reserved.
#include <string>
#include <vector>

namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

void genRandom(std::vector<float>& vec);
void genRandom(float* vec, size_t len);
void Print(float* vec, size_t len);
float Sum(float* vec, size_t len);

}  // namespace cudabm
