#include <doctest.h>
#include <core.h>

TEST_CASE("sample is one byte") {
    CHECK(sizeof(core::Sample) == 1);
}

TEST_CASE("blocksize 4x4 has width 4") {
    using namespace core;
    CHECK(blockWidth(BlockSize::BLOCK_4X4) == 4);
}

TEST_CASE("blocksize 8x8 has width 8") {
    using namespace core;
    CHECK(blockWidth(BlockSize::BLOCK_8X8) == 8);
}

TEST_CASE("blocksize 8x8 has height 8") {
    using namespace core;
    CHECK(blockHeight(BlockSize::BLOCK_8X8) == 8);
}

TEST_CASE("samples per pixel is 3 (yuv)") {
    CHECK(core::planeCount() == 3);
}
