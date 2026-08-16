#include "gpurt.h"

#include <nvrtc.h>

#include <stdexcept>

namespace gpurt {

namespace {

[[noreturn]] void throwCuda(const char* what, CUresult rc) {
    throw std::runtime_error(std::string(what) + " (CUresult " +
                             std::to_string(static_cast<int>(rc)) + ")");
}

[[noreturn]] void throwNvrtc(const char* what, nvrtcResult rc) {
    throw std::runtime_error(std::string(what) + " (NVRTC " +
                             std::to_string(static_cast<int>(rc)) + ")");
}

CUcontext context() {
    static const CUcontext ctx = []() {
        CUdevice dev = 0;
        CUcontext c = nullptr;
        if (cuInit(0) != CUDA_SUCCESS) {
            throw std::runtime_error("cuInit failed");
        }
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("cuDeviceGet failed");
        }
        if (cuCtxCreate(&c, CU_CTX_SCHED_AUTO, dev) != CUDA_SUCCESS) {
            throw std::runtime_error("cuCtxCreate failed");
        }
        return c;
    }();
    return ctx;
}

}  // namespace

int deviceCount() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return 0;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return 0;
    }
    return count;
}

GpuContext::GpuContext() {
    (void)context();
}

GpuContext::~GpuContext() = default;

DeviceBuffer::DeviceBuffer(std::size_t bytes) {
    (void)context();
    const CUresult rc = cuMemAlloc(&ptr_, bytes);
    if (rc != CUDA_SUCCESS) {
        throwCuda("cuMemAlloc failed", rc);
    }
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_ != 0) {
        cuMemFree(ptr_);
    }
}

void DeviceBuffer::uploadFrom(const void* host, std::size_t bytes) {
    const CUresult rc = cuMemcpyHtoD(ptr_, host, bytes);
    if (rc != CUDA_SUCCESS) {
        throwCuda("cuMemcpyHtoD failed", rc);
    }
}

void DeviceBuffer::downloadTo(void* host, std::size_t bytes) {
    const CUresult rc = cuMemcpyDtoH(host, ptr_, bytes);
    if (rc != CUDA_SUCCESS) {
        throwCuda("cuMemcpyDtoH failed", rc);
    }
}

Kernel::Kernel(const std::string& ptx, const std::string& entryName) {
    (void)context();
    CUresult rc = cuModuleLoadData(&module_, ptx.c_str());
    if (rc != CUDA_SUCCESS) {
        throwCuda("cuModuleLoadData failed", rc);
    }
    rc = cuModuleGetFunction(&fn_, module_, entryName.c_str());
    if (rc != CUDA_SUCCESS) {
        throwCuda("cuModuleGetFunction failed", rc);
    }
}

Kernel::~Kernel() {
    if (module_ != nullptr) {
        cuModuleUnload(module_);
    }
}

void Kernel::launch(unsigned gridX, unsigned gridY, unsigned blockX, unsigned blockY,
                    void** args) {
    const CUresult rc = cuLaunchKernel(fn_, gridX, gridY, 1, blockX, blockY, 1, 0, nullptr,
                                       args, nullptr);
    if (rc != CUDA_SUCCESS) {
        throwCuda("cuLaunchKernel failed", rc);
    }
    const CUresult sync = cuCtxSynchronize();
    if (sync != CUDA_SUCCESS) {
        throwCuda("cuCtxSynchronize failed", sync);
    }
}

std::vector<std::string> ptxEntryNames(const std::string& ptx) {
    std::vector<std::string> names;
    std::size_t pos = 0;
    const std::string marker = ".entry ";
    while ((pos = ptx.find(marker, pos)) != std::string::npos) {
        pos += marker.size();
        const std::size_t end = ptx.find('(', pos);
        if (end != std::string::npos && end > pos) {
            names.push_back(ptx.substr(pos, end - pos));
        }
    }
    return names;
}

std::optional<std::string> compileToPtx(const std::string& source, const std::string& arch) {
    nvrtcProgram prog{};
    if (nvrtcCreateProgram(&prog, source.c_str(), "kernel.cu", 0, nullptr, nullptr) !=
        NVRTC_SUCCESS) {
        return std::nullopt;
    }

    const std::string gpuArch = "--gpu-architecture=" + arch;
    const char* opts[] = {gpuArch.c_str()};
    const nvrtcResult rc = nvrtcCompileProgram(prog, 1, opts);
    if (rc != NVRTC_SUCCESS) {
        std::size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        if (logSize > 0) {
            nvrtcGetProgramLog(prog, log.data());
        }
        nvrtcDestroyProgram(&prog);
        (void)rc;
        fprintf(stderr, "NVRTC compile failed:\n%s\n", log.c_str());
        return std::nullopt;
    }

    std::size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0');
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    return ptx;
}

}  // namespace gpurt
