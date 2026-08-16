#pragma once

#include <cstdint>
#include <string>

namespace transforms {

std::int32_t cospi13(int index);

std::int32_t sinpi13(int index);

std::int32_t halfBtf(std::int32_t w0, std::int32_t in0, std::int32_t w1, std::int32_t in1, int bit);

std::int32_t roundShift(std::int64_t value, int bit);

void dct2Forward4x4(const float in[16], float out[16]);

void adst4Forward4x4(const float in[16], float out[16]);

std::string dct2Forward4x4CuSource();

std::string adst4Forward4x4CuSource();

}  // namespace transforms
