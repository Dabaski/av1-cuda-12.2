add_library(av1_intra STATIC intra.cpp)
target_include_directories(av1_intra INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(av1_intra INTERFACE av1_pixels av1_gpurt)

add_executable(av1_l4_intra_tests tests/test_harness.cpp tests/test_intra.cpp)
target_include_directories(av1_l4_intra_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(av1_l4_intra_tests PRIVATE av1_intra)
av1_cuda_test(l4_intra av1_l4_intra_tests)
