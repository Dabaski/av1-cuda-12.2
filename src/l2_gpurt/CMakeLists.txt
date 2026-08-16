find_package(CUDAToolkit QUIET)

add_library(av1_gpurt STATIC gpurt.cpp)
target_include_directories(av1_gpurt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(av1_gpurt PUBLIC av1_core)
if(CUDAToolkit_FOUND)
  target_link_libraries(av1_gpurt PUBLIC CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
endif()

add_executable(av1_l2_gpurt_tests tests/test_harness.cpp tests/test_gpurt.cpp)
target_include_directories(av1_l2_gpurt_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(av1_l2_gpurt_tests PRIVATE av1_gpurt)
av1_cuda_test(l2_gpurt av1_l2_gpurt_tests)
