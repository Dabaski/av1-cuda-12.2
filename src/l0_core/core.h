#pragma once

#include <cstdint>

namespace core {

using Sample = std::uint8_t;

enum class BlockSize {
    BLOCK_4X4 = 0,
    BLOCK_8X8 = 1,
    kBlockSizeCount,
};

inline int blockWidth(BlockSize bs) {
    return 4 << static_cast<int>(bs);
}

inline int blockHeight(BlockSize bs) {
    return 4 << static_cast<int>(bs);
}

inline int planeCount() {
    return 3;
}

}  // namespace core
