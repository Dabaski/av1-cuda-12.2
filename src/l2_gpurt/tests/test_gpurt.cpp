#include <doctest.h>
#include <gpurt.h>

TEST_CASE("nvrtc compiles a trivial kernel to ptx for pascal") {
    const std::string src =
        "extern \"C\" __global__ void add(const int* a, int* b, int n) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (i < n) b[i] = a[i] + 1;\n"
        "}\n";
    std::optional<std::string> ptx = gpurt::compileToPtx(src, "compute_61");
    REQUIRE(ptx.has_value());
}

TEST_CASE("ptx exposes one entry point for the add kernel") {
    const std::string src =
        "extern \"C\" __global__ void add(const int* a, int* b, int n) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (i < n) b[i] = a[i] + 1;\n"
        "}\n";
    const std::optional<std::string> ptx = gpurt::compileToPtx(src, "compute_61");
    REQUIRE(ptx.has_value());
    CHECK(gpurt::ptxEntryNames(*ptx).size() == 1);
}

TEST_CASE("add kernel executes on the gpu") {
    if (gpurt::deviceCount() == 0) {
        MESSAGE("SKIP: no CUDA device");
        return;
    }
    gpurt::GpuContext ctx;
    const int n = 4;
    const int inData[4] = {1, 2, 3, 4};
    int outData[4] = {0, 0, 0, 0};
    gpurt::DeviceBuffer inBuf(sizeof(inData));
    gpurt::DeviceBuffer outBuf(sizeof(outData));
    inBuf.uploadFrom(inData, sizeof(inData));

    const std::string src =
        "extern \"C\" __global__ void add(const int* a, int* b, int n) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (i < n) b[i] = a[i] + 1;\n"
        "}\n";
    const std::string ptx = *gpurt::compileToPtx(src, "compute_61");
    const std::string entry = gpurt::ptxEntryNames(ptx).at(0);
    gpurt::Kernel k(ptx, entry);

    CUdeviceptr din = inBuf.get();
    CUdeviceptr dout = outBuf.get();
    int dn = n;
    void* args[] = {&din, &dout, &dn};
    k.launch(1, 1, 4, 1, args);

    outBuf.downloadTo(outData, sizeof(outData));
    CHECK(outData[3] == 5);
}
