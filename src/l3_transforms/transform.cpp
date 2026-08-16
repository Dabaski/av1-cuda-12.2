#include "transform.h"

#include <cmath>

namespace transforms {

namespace {

// svt_aom_eb_av1_cospi_arr_data[3], the cos_bit=13 row
// (svt_aom_eb_av1_cospi_arr_data[i][j] = round(cos(M_PI*j/128) * (1<<(cos_bit_min+i))))
const std::int32_t kCospi13[64] = {
    8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946, 7895, 7839, 7779, 7713, 7643,
    7568, 7489, 7405, 7317, 7225, 7128, 7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933,
    5793, 5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038, 3862, 3683, 3503, 3320,
    3135, 2948, 2760, 2570, 2378, 2185, 1990, 1795, 1598, 1401, 1202, 1003, 803,  603,  402,  201,
};

// svt_aom_eb_av1_sinpi_arr_data[3], the cos_bit=13 row
// (svt_aom_eb_av1_sinpi_arr_data[i][j] = round((sqrt(2)*sin(j*Pi/9)*2/3)*(1<<(cos_bit_min+i))))
const std::int32_t kSinpi13[5] = {0, 2642, 4964, 6689, 7606};

}  // namespace

std::int32_t cospi13(int index) {
    return kCospi13[index];
}

std::int32_t sinpi13(int index) {
    return kSinpi13[index];
}

std::int32_t halfBtf(std::int32_t w0, std::int32_t in0, std::int32_t w1, std::int32_t in1, int bit) {
    const std::int64_t result64    = static_cast<std::int64_t>(w0 * in0) + static_cast<std::int64_t>(w1 * in1);
    const std::int64_t intermediate = result64 + (1LL << (bit - 1));
    return static_cast<std::int32_t>(intermediate >> bit);
}

std::int32_t roundShift(std::int64_t value, int bit) {
    return static_cast<std::int32_t>((value + (1LL << (bit - 1))) >> bit);
}

constexpr int kN = 4;

namespace {

double dctBasis(int k, int n) {
    return std::cos(3.14159265358979323846 * static_cast<double>(k) *
                    (2.0 * static_cast<double>(n) + 1.0) / (2.0 * static_cast<double>(kN)));
}

double adstBasis(int k, int n) {
    return std::sin(3.14159265358979323846 *
                    (2.0 * static_cast<double>(n) + 1.0) *
                    (2.0 * static_cast<double>(k) + 1.0) /
                    (4.0 * static_cast<double>(kN)));
}

}  // namespace

void adst4Forward4x4(const float in[16], float out[16]) {
    double rows[kN][kN] = {};
    for (int y = 0; y < kN; ++y) {
        for (int k = 0; k < kN; ++k) {
            double sum = 0.0;
            for (int n = 0; n < kN; ++n) {
                sum += static_cast<double>(in[y * kN + n]) * adstBasis(k, n);
            }
            rows[y][k] = sum;
        }
    }
    for (int k = 0; k < kN; ++k) {
        for (int l = 0; l < kN; ++l) {
            double sum = 0.0;
            for (int n = 0; n < kN; ++n) {
                sum += rows[n][l] * adstBasis(k, n);
            }
            out[k * kN + l] = static_cast<float>(sum);
        }
    }
}

std::string adst4Forward4x4CuSource() {
    return R"CUDA(
extern "C" __global__ void adst4_fwd_4x4(const float* in, float* out) {
    const int idx = threadIdx.x;
    const int k = idx >> 2;
    const int l = idx & 3;
    const float pi = 3.14159265358979323846f;
    float s = 0.0f;
    for (int n = 0; n < 4; ++n) {
        for (int m = 0; m < 4; ++m) {
            float ck = sinf(pi * (2 * n + 1) * (2 * k + 1) / 16.0f);
            float cl = sinf(pi * (2 * m + 1) * (2 * l + 1) / 16.0f);
            s += in[n * 4 + m] * ck * cl;
        }
    }
    out[idx] = s;
}
)CUDA";
}

std::string dct2Forward4x4CuSource() {
    return R"CUDA(
extern "C" __global__ void dct2_fwd_4x4(const float* in, float* out) {
    const int idx = threadIdx.x;
    const int k = idx >> 2;
    const int l = idx & 3;
    const float pi = 3.14159265358979323846f;
    float s = 0.0f;
    for (int n = 0; n < 4; ++n) {
        for (int m = 0; m < 4; ++m) {
            float ck = cosf(pi * k * (2 * n + 1) / 8.0f);
            float cl = cosf(pi * l * (2 * m + 1) / 8.0f);
            s += in[n * 4 + m] * ck * cl;
        }
    }
    out[idx] = s;
}
)CUDA";
}

void dct2Forward4x4(const float in[16], float out[16]) {
    double rows[kN][kN] = {};
    for (int y = 0; y < kN; ++y) {
        for (int k = 0; k < kN; ++k) {
            double sum = 0.0;
            for (int n = 0; n < kN; ++n) {
                sum += static_cast<double>(in[y * kN + n]) * dctBasis(k, n);
            }
            rows[y][k] = sum;
        }
    }
    for (int k = 0; k < kN; ++k) {
        for (int l = 0; l < kN; ++l) {
            double sum = 0.0;
            for (int n = 0; n < kN; ++n) {
                sum += rows[n][l] * dctBasis(k, n);
            }
            out[k * kN + l] = static_cast<float>(sum);
        }
    }
}

}  // namespace transforms
