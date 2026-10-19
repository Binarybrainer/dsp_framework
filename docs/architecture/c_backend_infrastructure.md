# ⚙️ C++ Backend Infrastructure Specification

To achieve real-time performance for stateful DSP blocks (like PLLs and CDRs), we must offload sample-by-sample feedback loops to C++. This document specifies the infrastructure to compile C++ extensions natively within the Python package using **CMake** and **pybind11**, and establishes a **zero-copy memory interface** for seamless NumPy integration.

## 1. Package Directory Structure

We use `scikit-build-core` as our PEP 517 build backend. This allows standard `pip install .` commands to automatically trigger CMake, compile the C++ code, and bundle the resulting `.so` / `.pyd` binaries alongside the Python code.

```text
dsp_framework/
│
├── pyproject.toml              # Build system configuration
├── CMakeLists.txt              # C++ Build instructions
│
├── csrc/                       # C++ Source Code
│   ├── include/
│   │   └── dsp_math.hpp        # C++ Headers
│   └── bindings.cpp            # pybind11 wrapper definitions
│
└── src/
    └── dsp_framework/          # Python Source Code
        ├── __init__.py
        ├── core/
        │   └── signal_buffer.py
        └── blocks/
            └── fast_pll.py     # Python wrapper calling the C++ binary
```

## 2. Build System Configuration

### 2.1. `pyproject.toml`
Define the build system to use `scikit-build-core` and ensure `pybind11` is available during the build phase.

```toml
[build-system]
requires = ["scikit-build-core>=0.4.3", "pybind11>=2.11"]
build-backend = "scikit_build_core.build"

[project]
name = "dsp_framework"
version = "0.1.0"
description = "High-performance Hybrid Python/C++ DSP Framework"
dependencies = ["numpy", "scipy"]
```

### 2.2. `CMakeLists.txt`
Configure CMake to locate `pybind11` and compile the `csrc` directory into a Python extension module named `dsp_core_cpp`.

```cmake
cmake_minimum_required(VERSION 3.15...3.27)
project(dsp_framework_cpp)

# Find pybind11 (provided by scikit-build-core/pip environment)
find_package(pybind11 CONFIG REQUIRED)

# Define the Python extension module
pybind11_add_module(dsp_core_cpp 
    csrc/bindings.cpp
)

# Optional: Enable C++20 and aggressive compiler optimizations
target_compile_features(dsp_core_cpp PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dsp_core_cpp PRIVATE -O3 -march=native -ffast-math)
endif()
```

## 3. The Zero-Copy Memory Interface

The most critical part of this infrastructure is preventing memory duplication. When Python passes a `SignalBuffer.data` (which is a NumPy array) to C++, we must access the exact same RAM addresses.

We achieve this using `pybind11::array_t` and `pybind11::buffer_info`.

### 3.1. C++ Implementation (`csrc/bindings.cpp`)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <stdexcept>

namespace py = pybind11;

// Define a unified complex type alias
using Complex32 = std::complex<float>;
using Complex64 = std::complex<double>;

/**
 * A generalized template function demonstrating zero-copy memory access.
 * Modifies the array in-place or writes to a pre-allocated output buffer.
 */
void process_fast_loop(py::array_t<Complex64> input_array, 
                       py::array_t<Complex64> output_array) {
    
    // 1. Request buffer information (Zero-copy access)
    py::buffer_info buf_in = input_array.request();
    py::buffer_info buf_out = output_array.request();

    // 2. Validate dimensions
    if (buf_in.ndim != 1 || buf_out.ndim != 1) {
        throw std::runtime_error("Input and Output must be 1-D NumPy arrays");
    }
    if (buf_in.size != buf_out.size) {
        throw std::runtime_error("Input and Output buffer sizes must match");
    }

    // 3. Extract raw C++ pointers
    auto* ptr_in = static_cast<Complex64*>(buf_in.ptr);
    auto* ptr_out = static_cast<Complex64*>(buf_out.ptr);
    size_t num_samples = buf_in.shape[0];

    // 4. Execute high-speed C++ loop (Bypassing Python GIL)
    for (size_t i = 0; i < num_samples; ++i) {
        // Example logic: Copy and rotate phase
        ptr_out[i] = ptr_in[i] * Complex64(0.0, 1.0); 
    }
}

// 5. Bind functions to the Python module
PYBIND11_MODULE(dsp_core_cpp, m) {
    m.doc() = "C++ backend core for dsp_framework";
    
    m.def("process_fast_loop", &process_fast_loop, 
          "A fast C++ loop operating on zero-copy NumPy arrays",
          py::arg("input_array"), py::arg("output_array"));
}
```

## 4. Python Wrapper Integration

Once compiled, the C++ module `dsp_core_cpp` can be imported directly into Python. To ensure memory safety, the Python `DSPBlock` pre-allocates the output buffer using `np.zeros_like()` before passing it down to C++.

```python
import numpy as np
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

# Import the compiled C++ extension
import dsp_core_cpp

class FastLoopBlock(DSPBlock):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        
    def reset(self) -> None:
        pass

    def process(self, input_buf: SignalBuffer, **kwargs) -> SignalBuffer:
        # 1. Ensure input data is contiguous and of the correct type (Float64 Complex)
        # np.ascontiguousarray ensures the C++ pointer can iterate linearly
        data_in = np.ascontiguousarray(input_buf.data, dtype=np.complex128)
        
        # 2. Pre-allocate the output array in Python to manage memory safely
        data_out = np.zeros_like(data_in)
        
        # 3. Call the zero-copy C++ function
        dsp_core_cpp.process_fast_loop(data_in, data_out)
        
        # 4. Wrap back into SignalBuffer
        return input_buf.clone_metadata(data_out)
```

By following this specification, your framework effortlessly bridges the gap between Python's rapid prototyping and C++'s bare-metal performance, keeping the memory entirely shared between both boundaries.