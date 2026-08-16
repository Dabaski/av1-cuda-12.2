add_library(av1_pixels INTERFACE)
target_include_directories(av1_pixels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(av1_pixels INTERFACE av1_core)

add_executable(av1_l1_pixels_tests tests/test_harness.cpp tests/test_pixels.cpp)
target_include_directories(av1_l1_pixels_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(av1_l1_pixels_tests PRIVATE av1_pixels)
add_test(NAME l1_pixels COMMAND av1_l1_pixels_tests)
