# 🛠️ Technical Documentation: CMake Build System (`CMakeLists.txt`)

This document provides a detailed breakdown of the CMake build configuration for the **Multi-Language DSP Simulation Framework**. It explains how the build system manages C++ compilation, shared library exports for Julia, Python extension modules via `pybind11`, and automated unit testing using `GoogleTest`.

---

## 1. Overview & Build Architecture

The CMake build script coordinates four core compilation targets:

| Target | Type | Purpose / Description |
| :--- | :--- | :--- |
| **`dsp_core`** | Static Library (`.a` / `.lib`) | High-performance C++20 core engine containing memory management and DSP algorithms. |
| **`dsp_abi`** | Shared Library (`.so` / `.dll` / `.dylib`) | Dynamic library exposing C-compatible ABI symbols for Julia (`ccall`) and standard C foreign function interfaces. |
| **`dsp_framework_py`** | Python C-Extension Module | `pybind11`-generated extension module allowing native Python imports with zero-copy array bindings. |
| **`dsp_tests`** | Executable | Test runner executing C++ unit tests via GoogleTest and CTest. |

---

## 2. Global Build Configuration & Compiler Flags

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

* **C++ Standard:** Enforces **C++20** strict compatibility across all compilers.
* **Optimization Flags:**
  * **GCC / Clang:** `-O3` (high optimization), `-march=native` (enables host-specific SIMD vectorization such as AVX2 / AVX-512), `-Wall -Wextra` (comprehensive warning checks).
  * **MSVC:** `/O2` (maximum speed optimization), `/W4` (high warning level), `/permissive-` (strict C++ conformance).

---

## 3. Configurable Build Options

You can toggle optional features during configuration using `-D<OPTION>=<ON|OFF>`:

```cmake
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" ON)
option(BUILD_TESTS "Build C++ unit tests using GoogleTest" ON)
```

* **`BUILD_PYTHON_BINDINGS`** *(Default: `ON`)*: Enables automatic detection/fetching of `pybind11` and builds the `dsp_framework_py` module.
* **`BUILD_TESTS`** *(Default: `ON`)*: Downloads GoogleTest if missing and compiles executable test suites under `tests/cpp/`.

---

## 4. Deep Dive into Build Targets

### 4.1 C++ Core Static Library (`dsp_core`)
Collects all C++ engine source files recursively from `src/core/` and `src/blocks/`:

```cmake
file(GLOB_RECURSE DSP_CORE_SOURCES "src/core/*.cpp" "src/blocks/*.cpp")
add_library(dsp_core STATIC ${DSP_CORE_SOURCES})
```

### 4.2 C-ABI Shared Library (`dsp_abi`)
Compiles a dynamic shared library required by Julia's `ccall` or standard C applications:

```cmake
add_library(dsp_abi SHARED ${DSP_CORE_SOURCES})
```

### 4.3 Python Extension (`dsp_framework_py`)
Employs `FetchContent` to dynamically retrieve `pybind11` (v2.11.1) if it is not installed locally on the host machine:

```cmake
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
```
Links `src/bindings/python/*.cpp` with `dsp_core` to output `dsp_framework_py.so` or `.pyd` directly into `build/lib/`.

### 4.4 C++ Unit Tests (`dsp_tests`)
Integrates GoogleTest (v1.14.0) via CMake's `FetchContent` module and enables `CTest` integration:

```cmake
gtest_discover_tests(dsp_tests)
```

---

## 5. Usage & Command Reference

### 5.1 Standard Build (All Targets Enabled)

```bash
# 1. Create build directory
mkdir build && cd build

# 2. Configure project
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. Compile in parallel
cmake --build . --parallel
```

### 5.2 Lightweight Build (C++ Only, No Python/Tests)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_PYTHON_BINDINGS=OFF \
         -DBUILD_TESTS=OFF
cmake --build .
```

### 5.3 Executing Unit Tests

```bash
cd build
ctest --output-on-failure
```

---

## 6. Output File Locations

Following compilation, artifacts are grouped in the `build/` directory:

```text
build/
├── bin/
│   └── dsp_tests             # Test executable
└── lib/
    ├── libdsp_core.a         # Static C++ core library
    ├── libdsp_abi.so / .dll  # Shared C-ABI library for Julia
    └── dsp_framework_py.so   # Python C-extension module
```