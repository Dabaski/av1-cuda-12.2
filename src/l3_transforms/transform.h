#pragma once

#include <cstdint>
#include <string>

namespace transforms {

std::int32_t cospi13(int index);

std::int32_t sinpi13(int index);

std::int32_t halfBtf(std::int32_t w0, std::int32_t in0, std::int32_t w1, std::int32_t in1, int bit);

std::int32_t roundShift(std::int64_t value, int bit);

void fdct4(const std::int32_t input[4], std::int32_t output[4]);

void fadst4(const std::int32_t input[4], std::int32_t output[4]);

enum class TxType {
    DCT_DCT,
    ADST_ADST,
};

void fwdTxfm2d4x4(const std::int16_t* input, std::int32_t* output, std::uint32_t stride, TxType type);

void dct2Forward4x4(const float in[16], float out[16]);

void adst4Forward4x4(const float in[16], float out[16]);

std::string dct2Forward4x4CuSource();

std::string adst4Forward4x4CuSource();

}  // namespace transforms
