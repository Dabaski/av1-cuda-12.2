add_library(av1_core INTERFACE)
target_include_directories(av1_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/third_party/doctest)

add_executable(av1_l0_core_tests tests/test_harness.cpp tests/test_core.cpp)
target_include_directories(av1_l0_core_tests PRIVATE ${PROJECT_SOURCE_DIR}/third_party/doctest)
target_link_libraries(av1_l0_core_tests PRIVATE av1_core)
add_test(NAME l0_core COMMAND av1_l0_core_tests)
