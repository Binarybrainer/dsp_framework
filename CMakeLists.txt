cmake_minimum_required(VERSION 3.20)
project(DSPFramework 
    VERSION 0.1.0 
    DESCRIPTION "High-Performance Multi-Language DSP Simulation Framework"
    LANGUAGES CXX C
)

# -----------------------------------------------------------------------------
# Global C++ Standard & Build Options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)
option(BUILD_TESTS "Build C++ unit tests using GoogleTest" ON)

# Optimization & Compiler Flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------------
# Include Directories
# -----------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -----------------------------------------------------------------------------
# 1. C++ Core Static Library (dsp_core)
# -----------------------------------------------------------------------------
file(GLOB_RECURSE DSP_CORE_SOURCES 
    "src/core/*.cpp"
    "src/blocks/*.cpp"
)

add_library(dsp_core STATIC ${DSP_CORE_SOURCES})

target_include_directories(dsp_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------------------------------------------------------
# 2. C-ABI Shared Library (dsp_abi) for Julia & C Interop
# -----------------------------------------------------------------------------
add_library(dsp_abi SHARED ${DSP_CORE_SOURCES})

target_include_directories(dsp_abi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(dsp_abi PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "dsp_abi"
)

# -----------------------------------------------------------------------------
# 3. Python Extension Module via pybind11 (Optional)
# -----------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found via CMake config, trying FetchContent...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    file(GLOB_RECURSE PYTHON_BINDING_SOURCES "src/bindings/python/*.cpp")

    pybind11_add_module(dsp_framework_py ${PYTHON_BINDING_SOURCES})
    target_link_libraries(dsp_framework_py PRIVATE dsp_core)
    
    set_target_properties(dsp_framework_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# -----------------------------------------------------------------------------
# 4. Unit Testing (GoogleTest via CTest)
# -----------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    file(GLOB_RECURSE CPP_TEST_SOURCES "tests/cpp/*.cpp")

    if(CPP_TEST_SOURCES)
        add_executable(dsp_tests ${CPP_TEST_SOURCES})
        target_link_libraries(dsp_tests PRIVATE dsp_core gtest_main)
        
        include(GoogleTest)
        gtest_discover_tests(dsp_tests)
    endif()
endif()