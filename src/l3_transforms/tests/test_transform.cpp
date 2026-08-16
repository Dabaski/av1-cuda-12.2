#include <doctest.h>
#include <cmath>
#include <gpurt.h>
#include <transform.h>

TEST_CASE("cospi cos_bit=13 index 16 is 7568") {
    // golden: svt_aom_eb_av1_cospi_arr_data[13 - 10][16]
    CHECK(transforms::cospi13(16) == 7568);
}

TEST_CASE("cospi cos_bit=13 index 32 is 5793") {
    // golden: svt_aom_eb_av1_cospi_arr_data[13 - 10][32]
    CHECK(transforms::cospi13(32) == 5793);
}

TEST_CASE("cospi cos_bit=13 index 48 is 3135") {
    // golden: svt_aom_eb_av1_cospi_arr_data[13 - 10][48]
    CHECK(transforms::cospi13(48) == 3135);
}

TEST_CASE("sinpi cos_bit=13 index 4 is 7606") {
    // golden: svt_aom_eb_av1_sinpi_arr_data[13 - 10][4]
    CHECK(transforms::sinpi13(4) == 7606);
}

TEST_CASE("half_btf rounds 5793*8+5793*8 at bit 13 to 11") {
    CHECK(transforms::halfBtf(5793, 8, 5793, 8, 13) == 11);
}

TEST_CASE("round_shift rounds 96784 by 13 bits to 12") {
    CHECK(transforms::roundShift(96784, 13) == 12);
}

TEST_CASE("fdct4 matches svt_av1_fdct4_new golden") {
    // golden: svt_av1_fdct4_new @ cos_bit=13, input {5, 3, 7, 1}
    const std::int32_t in[4] = {5, 3, 7, 1};
    const std::int32_t out[4] = {11, 2, -3, 5};
    std::int32_t got[4] = {0};
    transforms::fdct4(in, got);
    CHECK(got[0] == out[0]);
}

TEST_CASE("fdct4 golden output 1 is 2") {
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fdct4(in, got);
    CHECK(got[1] == 2);
}

TEST_CASE("fdct4 golden output 2 is -3") {
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fdct4(in, got);
    CHECK(got[2] == -3);
}

TEST_CASE("fdct4 golden output 3 is 5") {
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fdct4(in, got);
    CHECK(got[3] == 5);
}

TEST_CASE("fadst4 golden output 0 is 10") {
    // golden: svt_av1_fadst4_new @ cos_bit=13, input {5, 3, 7, 1}
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fadst4(in, got);
    CHECK(got[0] == 10);
}

TEST_CASE("fadst4 golden output 1 is 6") {
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fadst4(in, got);
    CHECK(got[1] == 6);
}

TEST_CASE("fadst4 golden output 2 is -1") {
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fadst4(in, got);
    CHECK(got[2] == -1);
}

TEST_CASE("fadst4 golden output 3 is 6") {
    const std::int32_t in[4] = {5, 3, 7, 1};
    std::int32_t got[4] = {0};
    transforms::fadst4(in, got);
    CHECK(got[3] == 6);
}

TEST_CASE("fadst4 all-zero input early-outs to zeros") {
    const std::int32_t in[4] = {0, 0, 0, 0};
    std::int32_t got[4] = {42, 42, 42, 42};
    transforms::fadst4(in, got);
    CHECK(got[2] == 0);
}

TEST_CASE("forward 4x4 dct of a constant block is all dc") {
    const float in[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float out[16] = {0};
    transforms::dct2Forward4x4(in, out);
    CHECK(out[0] == doctest::Approx(16.0f));
}

TEST_CASE("forward 4x4 adst of a constant block has known dc") {
    const float in[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float out[16] = {0};
    transforms::adst4Forward4x4(in, out);
    CHECK(out[0] == doctest::Approx(6.5685f).epsilon(1e-3));
}

TEST_CASE("gpu 4x4 adst matches host reference") {
    if (gpurt::deviceCount() == 0) {
        MESSAGE("SKIP: no CUDA device");
        return;
    }
    gpurt::GpuContext ctx;

    float in[16] = {9, 2, 3, 1, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 9, 1};
    float ref[16] = {0};
    transforms::adst4Forward4x4(in, ref);

    const std::string ptx = *gpurt::compileToPtx(transforms::adst4Forward4x4CuSource(), "compute_61");
    const std::string entry = gpurt::ptxEntryNames(ptx).at(0);
    gpurt::Kernel k(ptx, entry);

    gpurt::DeviceBuffer dIn(sizeof(in));
    gpurt::DeviceBuffer dOut(sizeof(ref));
    dIn.uploadFrom(in, sizeof(in));

    CUdeviceptr pin = dIn.get();
    CUdeviceptr pout = dOut.get();
    void* args[] = {&pin, &pout};
    k.launch(1, 1, 16, 1, args);

    float got[16] = {0};
    dOut.downloadTo(got, sizeof(got));

    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        if (std::abs(got[i] - ref[i]) > 1e-4f) {
            ok = false;
        }
    }
    CHECK(ok);
}

TEST_CASE("gpu 4x4 dct matches host reference") {
    if (gpurt::deviceCount() == 0) {
        MESSAGE("SKIP: no CUDA device");
        return;
    }
    gpurt::GpuContext ctx;

    float in[16] = {1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1};
    float ref[16] = {0};
    transforms::dct2Forward4x4(in, ref);

    const std::string ptx = *gpurt::compileToPtx(transforms::dct2Forward4x4CuSource(), "compute_61");
    const std::string entry = gpurt::ptxEntryNames(ptx).at(0);
    gpurt::Kernel k(ptx, entry);

    gpurt::DeviceBuffer dIn(sizeof(in));
    gpurt::DeviceBuffer dOut(sizeof(ref));
    dIn.uploadFrom(in, sizeof(in));

    CUdeviceptr pin = dIn.get();
    CUdeviceptr pout = dOut.get();
    void* args[] = {&pin, &pout};
    k.launch(1, 1, 16, 1, args);

    float got[16] = {0};
    dOut.downloadTo(got, sizeof(got));

    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        if (std::abs(got[i] - ref[i]) > 1e-4f) {
            ok = false;
        }
    }
    CHECK(ok);
}
