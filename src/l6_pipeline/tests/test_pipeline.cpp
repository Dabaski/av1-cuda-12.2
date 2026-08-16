#include <doctest.h>
#include <pipeline.h>

TEST_CASE("pipeline v-mode coeffs match svt composition golden") {
    // golden: build_intra_predictors (V_PRED) -> subtract -> svt_av1_fdct4_new
    // 2D @ cos_bit=13, src {21,3,5,9,9,11,3,7,7,13,5,1,15,4,25,2}, above {10,40,30,20}
    const std::int32_t golden[16] = {-520, 140, 324, 202, -17, 18, 102, -68,
                                     56,   3,   36,  120, -18, 23, 6,   -22};
    const std::uint8_t srcData[16] = {21, 3, 5, 9, 9, 11, 3, 7, 7, 13, 5, 1, 15, 4, 25, 2};
    const std::uint8_t above[4] = {10, 40, 30, 20};

    pixels::Plane plane(4, 4, 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            plane.at(x, y) = srcData[y * 4 + x];
        }
    }

    std::int32_t coeffs[16] = {0};
    pipeline::encodeBlock4x4(plane, 0, 0, above, 4, 0, nullptr, 0, 0, 0, intra::V_PRED, 0,
                             transforms::TxType::DCT_DCT, coeffs);

    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        if (coeffs[i] != golden[i]) {
            ok = false;
        }
    }
    CHECK(ok);
}

TEST_CASE("pipeline dc-mode coeffs match svt composition golden") {
    // golden: build_intra_predictors (DC_PRED) -> subtract ->
    // svt_av1_fdct4_new 2D @ cos_bit=13, src {9,4,7,5,12,8,3,6,15,2,11,4,6,9,13,2},
    // above {12,24,36,48}, left {6,18,30,42}
    const std::int32_t golden[16] = {-631, 53, 4,  55,  -16, 2,  45, -26,
                                     -12,  -27, -47, -2,  2,   -2, 16, 53};
    const std::uint8_t srcData[16] = {9, 4, 7, 5, 12, 8, 3, 6, 15, 2, 11, 4, 6, 9, 13, 2};
    const std::uint8_t above[4] = {12, 24, 36, 48};
    const std::uint8_t left[4] = {6, 18, 30, 42};

    pixels::Plane plane(4, 4, 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            plane.at(x, y) = srcData[y * 4 + x];
        }
    }

    std::int32_t coeffs[16] = {0};
    pipeline::encodeBlock4x4(plane, 0, 0, above, 4, 0, left, 4, 0, 0, intra::DC_PRED, 0,
                             transforms::TxType::DCT_DCT, coeffs);

    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        if (coeffs[i] != golden[i]) {
            ok = false;
        }
    }
    CHECK(ok);
}