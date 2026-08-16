#include <doctest.h>
#include <cmath>
#include <gpurt.h>
#include <transform.h>

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
