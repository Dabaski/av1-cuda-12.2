#pragma once

#include <cuda.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace gpurt {

std::optional<std::string> compileToPtx(const std::string& source, const std::string& arch);

std::vector<std::string> ptxEntryNames(const std::string& ptx);

int deviceCount();

class GpuContext {
public:
    GpuContext();
    ~GpuContext();
    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void uploadFrom(const void* host, std::size_t bytes);
    void downloadTo(void* host, std::size_t bytes);
    CUdeviceptr get() const { return ptr_; }

private:
    CUdeviceptr ptr_ = 0;
};

class Kernel {
public:
    Kernel(const std::string& ptx, const std::string& entryName);
    ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    void launch(unsigned gridX, unsigned gridY, unsigned blockX, unsigned blockY,
                void** args);

private:
    CUmodule module_ = nullptr;
    CUfunction fn_ = nullptr;
};

}  // namespace gpurt
