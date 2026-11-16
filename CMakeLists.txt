cmake_minimum_required(VERSION 3.18)
project(vaani_nlu VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PROJECT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(PROJECT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")

find_package(Python3 COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

add_library(normalizer SHARED "${PROJECT_SRC_DIR}/normalizer.cpp")
target_include_directories(normalizer PRIVATE "${PROJECT_INCLUDE_DIR}")
set_target_properties(normalizer PROPERTIES
    OUTPUT_NAME normalizer
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

add_library(vaani_runtime SHARED "${PROJECT_SRC_DIR}/runtime.cpp")
target_include_directories(vaani_runtime PRIVATE "${PROJECT_INCLUDE_DIR}" "${PROJECT_SRC_DIR}")
target_link_libraries(vaani_runtime PRIVATE normalizer)
target_compile_definitions(vaani_runtime PRIVATE
    VAANI_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/scripts/onnx_inference.py"
)

add_executable(vaani_cli "${PROJECT_SRC_DIR}/cli.cpp")
target_include_directories(vaani_cli PRIVATE "${PROJECT_INCLUDE_DIR}" "${PROJECT_SRC_DIR}")
target_link_libraries(vaani_cli PRIVATE vaani_runtime normalizer)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

if(Python3_Interpreter_FOUND)
    add_test(
        NAME e2e_test
        COMMAND "${Python3_EXECUTABLE}" "${PROJECT_TESTS_DIR}/test_e2e.py"
    )
    set_tests_properties(e2e_test PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        ENVIRONMENT "VAANI_LIB_PATH=$<TARGET_FILE:vaani_runtime>;LD_LIBRARY_PATH=$<TARGET_FILE_DIR:vaani_runtime>"
        TIMEOUT 600
    )
else()
    message(WARNING "Python 3 not found: the E2E test target will not be registered")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS normalizer vaani_runtime
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)
install(TARGETS vaani_cli RUNTIME DESTINATION bin)
install(FILES "${PROJECT_INCLUDE_DIR}/vaani_nlu.h" DESTINATION include)

message(STATUS "vaani_nlu ${PROJECT_VERSION} configured")
message(STATUS "  build/libnormalizer.so")
message(STATUS "  build/libvaani_runtime.so")
message(STATUS "  build/vaani_cli")
message(STATUS "  ctest --output-on-failure")
