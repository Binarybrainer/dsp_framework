# 📁 Technical Specification: Project Directory Architecture

This document defines the standard repository layout, source code organization, and build system integration for the multi-language (**C++20**, **Python**, **Julia**) DSP Simulation Framework.

---

## 1. Directory Tree Overview

```text
dsp-framework/
├── CMakeLists.txt                  # Root CMake build configuration
├── LICENSE                         # Project license
├── README.md                       # High-level documentation & quickstart
├── pyproject.toml                  # Python package & build metadata
├── Project.toml                    # Julia package configuration
│
├── cmake/                          # Custom CMake modules & toolchains
│   ├── FindJulia.cmake
│   └── PreventInSourceBuilds.cmake
│
├── include/                        # Public C++ headers (C-ABI & C++ Core)
│   └── dsp/
│       ├── abi/
│       │   └── dsp_abi.h           # C-compatible structs (SignalBuffer, FrameTensor)
│       ├── core/
│       │   ├── SignalBuffer.hpp
│       │   ├── FrameTensor.hpp
│       │   └── DSPModule.hpp
│       ├── blocks/
│       │   ├── pll.hpp             # PLL core definitions
│       │   ├── cdr.hpp             # CDR core definitions
│       │   └── mimo.hpp            # MIMO core definitions
│       └── utils/
│           └── aligned_alloc.hpp
│
├── src/                            # C++ Implementation source files
│   ├── core/
│   │   ├── SignalBuffer.cpp
│   │   └── PipelineEngine.cpp
│   ├── blocks/
│   │   ├── pll.cpp
│   │   ├── cdr.cpp
│   │   └── mimo.cpp
│   └── bindings/
│       └── python/
│           ├── pybind_main.cpp    # pybind11 module initialization
│           ├── bind_buffers.cpp   # Zero-copy buffer bindings
│           └── bind_blocks.cpp    # DSP module bindings
│
├── python/                         # Python wrapper package & high-level API
│   └── dsp_framework/
│       ├── __init__.py
│       ├── core.py                 # High-level Python wrappers around C++ extension
│       ├── visualization.py        # Matplotlib/Plotly plotting helpers (Eye, Constellation)
│       └── py.typed                 # PEP 561 type annotation flag
│
├── julia/                          # Julia package wrapper
│   ├── src/
│   │   ├── DSPFramework.jl         # Main Julia module
│   │   ├── abi_bindings.jl         # CFFI / CxxWrap wrappers
│   │   └── visualization.jl        # Plots.jl / Makie integration
│   └── test/
│       └── runtests.jl
│
├── tests/                          # C++ and Cross-Language Tests
│   ├── cpp/                        # Catch2 / GoogleTest unit tests
│   │   ├── test_buffer.cpp
│   │   └── test_pll.cpp
│   ├── python/                     # pytest suite for Python bindings
│   │   ├── test_zero_copy.py
│   │   └── test_cdr_pipeline.py
│   └── parity/                     # Multi-language numerical parity tests
│       └── test_cross_lang.py
│
├── docs/                           # Documentation sources
│   ├── architecture/
│   │   ├── memory_spec.md
│   │   └── repo_structure.md
│   └── api/
│
└── examples/                       # End-to-end simulation examples
    ├── cpp/
    │   └── serdes_link.cpp
    ├── python/
    │   ├── cdr_eye_diagram.ipynb
    │   └── mimo_ofdm_simulation.py
    └── julia/
        └── pll_transient_analysis.jl
```

---

## 2. Directory Responsibilities & Rules

### 2.1 C++ Core (`include/`, `src/`)
* **Header-Only vs Compiled:** Core template data structures (e.g., `SignalBuffer<T>`) reside under `include/dsp/core/`. Non-templated heavy logic (e.g., matrix solves in MIMO, state machines in CDR) must have headers in `include/dsp/` and implementation files in `src/`.
* **C-ABI Separation:** Standard C-compatible header files (such as `dsp_abi.h`) MUST remain free of C++ constructs to allow direct `ccall` or CFFI loading without C++ name mangling issues.

### 2.2 Python Package (`python/dsp_framework/`)
* Contains pure Python modules, type hints, and user-facing high-level wrappers.
* The compiled C++ binary extension (`_dsp_core.so` or `_dsp_core.pyd`) is placed directly inside `python/dsp_framework/` during build time.

### 2.3 Julia Package (`julia/`)
* Structured according to standard Julia package layouts (`src/` and `Project.toml`).
* Loads the compiled shared library (`libdsp_core.so`/`.dll`) using `Libdl` or `CxxWrap`.

### 2.4 Test Suite (`tests/`)
* **`tests/cpp/`:** Native C++ unit testing framework (GoogleTest / Catch2).
* **`tests/python/`:** Python test cases executed via `pytest`.
* **`tests/parity/`:** Integration tests verifying that given identical inputs, Python, Julia, and C++ routines yield numerically equivalent results within defined tolerances ($10^{-6}$).

---

## 3. Build & Packaging Integration

| Language | Build Manager | Artifact / Package Output |
| :--- | :--- | :--- |
| **C++ Engine** | CMake ($\ge$ 3.20) | `libdsp_core.so` / `dsp_core.dll` |
| **Python** | `setuptools` + `scikit-build-core` / `pybind11` | Wheel package (`.whl`) |
| **Julia** | `Pkg.jl` + `BinaryBuilder` / `Libdl` | Julia Package |

---

## 4. Guidelines for Adding New DSP Blocks

When adding a new DSP module (e.g., a new CDR or PLL filter):

1. **C++ Core Header:** Define module interface in `include/dsp/blocks/<module_name>.hpp`.
2. **C++ Implementation:** Implement algorithms in `src/blocks/<module_name>.cpp`.
3. **Python Binding:** Register class/functions in `src/bindings/python/bind_blocks.cpp`.
4. **Julia Binding:** Expose C-ABI or CxxWrap interfaces in `julia/src/abi_bindings.jl`.
5. **Unit Tests:** Add corresponding test cases in `tests/cpp/` and `tests/python/`.