#pragma once

#include <string>

namespace transforms {

void dct2Forward4x4(const float in[16], float out[16]);

void adst4Forward4x4(const float in[16], float out[16]);

std::string dct2Forward4x4CuSource();

std::string adst4Forward4x4CuSource();

}  // namespace transforms
