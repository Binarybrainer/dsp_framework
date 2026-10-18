# 🛠️ CMake Build System & Integration Guide

This guide provides complete documentation for configuring, building, and integrating the multi-language (**C++**, **Python**, and **Julia**) DSP Simulation Framework using `CMake`.

---

## 1. Overview & Build Options

The framework uses CMake ($\ge 3.20$) as its central build tool. The build system supports modular compilation, allowing you to compile only the core library or build Python extensions and unit tests as needed.

### Key Configuration Flags

| CMake Option | Default Value | Description |
| :--- | :--- | :--- |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Enables building the `pybind11` C-extension module for Python. |
| `BUILD_TESTS` | `ON` | Compiles C++ unit tests using GoogleTest/CTest. |
| `CMAKE_BUILD_TYPE` | `Release` | Build mode (`Release` for `-O3` optimizations, `Debug` for symbol debugging). |

---

## 2. Prerequisites & Environment Setup

Before building, ensure you have the required toolchains installed:

* **C++ Compiler:** Supporting C++20 (GCC $\ge 10$, Clang $\ge 11$, or MSVC 2019+).
* **CMake:** Version 3.20 or higher.
* **Python (Optional):** Python 3.8+ with `pybind11` installed (`pip install pybind11`).
* **Julia (Optional):** Julia 1.6+ for C-ABI shared library calls.

---

## 3. Step-by-Step Build Instructions

### 3.1 Standard C++ Core & C-ABI Build

To compile the C++ core library (`libdsp_core`) and the C-ABI shared library (`libdsp_abi`):

```bash
# 1. Create and enter the build directory
mkdir build && cd build

# 2. Configure the project with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 3. Compile using parallel build threads
cmake --build . --parallel
```

Upon successful build, the output libraries will be located in:
* **Linux/macOS:** `build/lib/libdsp_core.a` and `build/lib/libdsp_abi.so` (or `.dylib`)
* **Windows:** `build/bin/Release/dsp_core.lib` and `build/bin/Release/dsp_abi.dll`

---

### 3.2 Building with Python Bindings Enabled

To compile the Python module (`dsp_framework_py`):

1. Ensure `pybind11` is available in your active Python environment:
   ```bash
   pip install pybind11
   ```

2. Run CMake with the `BUILD_PYTHON_BINDINGS` flag set to `ON`:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
   cmake --build . --parallel
   ```

3. **Verifying Python Import:**
   You can add the compiled library output directory (`build/lib` or build folder) to your Python path:
   ```python
   import sys
   sys.path.append("./build/lib")
   import dsp_framework_py

   # Verify instantiation
   buf = dsp_framework_py.SignalBuffer(1024, 1e6)
   print("SignalBuffer created successfully!")
   ```

---

### 3.3 Running C++ Unit Tests

When `BUILD_TESTS=ON` (default), unit tests are built using GoogleTest. You can execute them directly with `ctest`:

```bash
# Run tests from the build directory
ctest --output-on-failure
```

---

## 4. Integration Details by Language

### 4.1 C++ Integration

In your external C++ projects, link against `dsp_core` and include the header path:

```cmake
target_link_libraries(your_app PRIVATE dsp_core)
target_include_directories(your_app PRIVATE path/to/include)
```

### 4.2 Julia Integration (C-ABI Shared Library)

Julia interacts with the framework via the C-ABI shared library using `ccall` or `Libdl`.

```julia
using Libdl

# Path to compiled shared library
const LIB_DSP = joinpath(@__DIR__, "build/lib/libdsp_abi.so")

# Example C-ABI wrapper function in Julia
function create_buffer(capacity::Int, sample_rate::Float64)
    return ccall((:dsp_create_buffer, LIB_DSP), Ptr{Cvoid}, (Csize_t, Cdouble), capacity, sample_rate)
end
```

---

## 5. Troubleshooting & Clean Builds

* **Force a clean build:**
  ```bash
  rm -rf build/
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  ```

* **Python environment mismatch:**
  If CMake detects the wrong Python binary, pass the explicit Python path:
  ```bash
  cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPython_EXECUTABLE=$(which python3)
  ```