#include "transform.h"

#include <cmath>

namespace transforms {

namespace {

constexpr int kN = 4;

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
