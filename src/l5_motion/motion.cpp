#include "motion.h"

namespace motion {

std::uint32_t sad8x8(const std::uint8_t* src, std::uint32_t srcStride, const std::uint8_t* ref,
                     std::uint32_t refStride) {
    std::uint32_t sadBlock = 0;

    for (std::uint32_t r = 0; r < 8; ++r) {
        for (std::uint32_t c = 0; c < 8; ++c) {
            const int d = (int)src[c] - (int)ref[c];
            sadBlock += (std::uint32_t)(d < 0 ? -d : d);
        }
        src += srcStride;
        ref += refStride;
    }

    return sadBlock;
}

std::string sad8x8CuSource() {
    return R"CUDA(
extern "C" __global__ void sad8x8_kernel(const unsigned char* src, const int* srcStride,
                                         const unsigned char* ref, const int* refStride, int* out) {
    int sad = 0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int d = (int)src[c] - (int)ref[c];
            if (d < 0) {
                d = -d;
            }
            sad += d;
        }
        src += *srcStride;
        ref += *refStride;
    }
    out[0] = sad;
}
)CUDA";
}

}  // namespace motion