#pragma once

#include <cstdint>
#include <string>

namespace motion {

std::uint32_t sad8x8(const std::uint8_t* src, std::uint32_t srcStride, const std::uint8_t* ref,
                     std::uint32_t refStride);

std::string sad8x8CuSource();

}  // namespace motion