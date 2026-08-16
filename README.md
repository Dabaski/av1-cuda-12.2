# AV1 CUDA 12.2 Port

GPU-accelerated port of SVT-AV1 transform/quantization paths to CUDA 12.2.

> **Disclaimer:** This project is actively being worked on. Code is incomplete,
> tests may be red, and APIs will change. Expect rough edges. Not ready for
> production use.

## Status

Work-in-progress CUDA implementation of AV1 encoder building blocks
(DCT/ADST forward transforms, quantization, block-level kernels) layered
over SVT-AV1's host reference code. See `AGENTS.md` for the development
methodology (incremental TDD) being followed.

## Layout

```
src/
  l0_core/        core types, test harness
  l1_pixels/      pixel operations
  l2_gpurt/       CUDA runtime wrappers
  l3_transforms/  forward/inverse transforms
  ...
CMakeLists.txt    top-level build
AGENTS.md         contribution / methodology rules
```

## Build

CUDA Toolkit 12.2 required. Standard CMake flow:

```
cmake -S . -B build
cmake --build build
ctest --test-dir build
```