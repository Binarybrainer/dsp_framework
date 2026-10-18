# 📡 Multi-Language DSP Simulation Framework

A high-performance, modular Digital Signal Processing (DSP) simulation framework designed for modeling complex communication systems such as **MIMO, CDR, PLL**,SerDes links, and multi-antenna processing. 

Built with a modern C++ core engine and native bindings, it enables **zero-copy high-speed array sharing** across **Python (NumPy)** and **Julia**.

---

## 🚀 Key Features

* **Zero-Copy Interoperability:** Memory allocated in C++ can be directly wrapped as NumPy arrays in Python or native Julia vectors without copy overhead.
* **Modern C++ Engine:** Fast, SIMD-aligned (64-byte) processing core targeting modern architectures (C++20).
* **Multi-Language Support:** Scripting in Python for fast prototyping/visualization, Julia for numerical modeling, or C++ for real-time execution.
* **Rich Block Library:** Built-in modules for phase-locked loops (PLL), clock and data recovery (CDR), MIMO matrix channels, precoders, and equalizers.
* **Comprehensive Test & Build Tooling:** Integrated CMake build system, GoogleTest suite, and `pytest` harnesses.

---

## 📁 Repository Architecture

```text
dsp-framework/
├── CMakeLists.txt                  # Central CMake build configuration
├── README.md                       # Main documentation & setup guide
├── pyproject.toml                  # Python package configuration
├── Project.toml                    # Julia package configuration
│
├── include/                        # C++ Public Headers
│   └── dsp/
│       ├── abi/                    # Stable C-ABI headers (dsp_abi.h)
│       ├── core/                   # SignalBuffer, FrameTensor, DSPModule
│       └── blocks/                 # PLL, CDR, MIMO, Gain, Filters
│
├── src/                            # C++ Implementations
│   ├── core/                       # Core memory engines & pipelines
│   ├── blocks/                     # Algorithmic block implementations
│   └── bindings/                   # Language bindings (pybind11 / C-ABI exports)
│
├── python/                         # Pure Python package wrappers
│   └── dsp_framework/
│
├── julia/                          # Julia package wrappers
│   └── src/
│
├── tests/                          # Automated Tests
│   ├── cpp/                        # GoogleTest suite
│   ├── python/                     # Pytest suite
│   └── parity/                     # Multi-language numerical parity tests
│
└── examples/                       # End-to-end simulation pipelines
```

---

## 🛠️ Building & Installation

### Prerequisites
* **C++ Compiler:** GCC $\ge 10$, Clang $\ge 11$, or MSVC 2019+ (C++20 support required).
* **CMake:** Version 3.20 or higher.
* **Python (Optional):** Python 3.8+ with `pybind11` installed (`pip install pybind11`).
* **Julia (Optional):** Julia 1.6+.

### 1. Build C++ Core & Shared Libraries

```bash
# Create build directory
mkdir build && cd build

# Configure CMake in Release mode
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build targets in parallel
cmake --build . --parallel
```

### 2. Build with Python Bindings Enabled

```bash
# Install pybind11 inside your active Python environment
pip install pybind11

# Enable Python bindings in CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --parallel
```

---

## 💡 Quickstart Examples

### Python Example
```python
import sys
sys.path.append("./build/lib")
import dsp_framework_py as dsp
import numpy as np

# Instantiate C++ Gain Block
gain_block = dsp.GainBlock(3.0)

# Create input buffer
in_signal = np.array([1.0 + 1.0j, 2.0 - 0.5j], dtype=np.complex64)
out_signal = np.zeros_like(in_signal)

# Process in place via zero-copy buffer sharing
gain_block.process(in_signal, out_signal)
print("Output:", out_signal)
```

### Julia Example
```julia
using Libdl

# Path to compiled C-ABI shared library
const LIB_DSP = joinpath(@__DIR__, "build/lib/libdsp_abi.so")

# Wrap native allocated C++ memory into Julia array
# (See julia/src/abi_bindings.jl for detailed functions)
```

### C++ Example
```cpp
#include "dsp/blocks/gain_block.hpp"
#include <iostream>

int main() {
    dsp::blocks::GainBlock gain(2.5f);
    std::vector<std::complex<float>> in = {{1.0f, 2.0f}};
    std::vector<std::complex<float>> out(1);

    gain.process(in.data(), out.data(), in.size());
    std::cout << "Processed: " << out[0] << std::endl;
    return 0;
}
```

---

## 🧩 Adding New DSP Blocks

To add a new block `<module_name>` to the framework, follow these 5 steps:

1. **C++ Header:** Define the class interface in `include/dsp/blocks/<module_name>.hpp`.
2. **C++ Source:** Implement algorithms in `src/blocks/<module_name>.cpp`.
3. **Python Binding:** Register class and methods in `src/bindings/python/bind_blocks.cpp`.
4. **Julia Binding:** Expose C-ABI interfaces in `julia/src/abi_bindings.jl`.
5. **Unit Tests:** Add test cases under `tests/cpp/` and `tests/python/`.

*For detailed instructions, refer to `adding_new_dsp_block_guide.md`.*

---

## 🧪 Running Tests

```bash
# Run C++ Unit Tests via CTest
cd build
ctest --output-on-failure

# Run Python pytest suite
pytest tests/python/
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.