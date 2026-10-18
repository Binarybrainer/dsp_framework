# 📐 Technical Specification: Unified Zero-Copy Memory Architecture

This document provides the design, memory layout, and implementation guidelines for `SignalBuffer` and `FrameTensor`, enabling **zero-copy memory sharing** across **C++20**, **Python (NumPy)**, and **Julia**.

---

## 1. Overview & Key Objectives

To enable high-performance DSP simulation across three language runtimes without data copying overhead:
- **Single Source of Truth:** Memory is allocated once (typically in C++ or through standard dynamic allocation) and wrapped seamlessly in Python and Julia.
- **C-ABI Stability:** Core data descriptors use a standard C-compatible `struct` to ensure direct interop via foreign function interfaces (FFI).
- **Contiguous & Aligned Memory:** Guarantees SIMD vectorization readiness (AVX-512 / ARM Neon) with 64-byte alignment.
- **Explicit Lifetime & Ownership Management:** Reference counting or custom deleter callbacks prevent premature deallocations or memory leaks across runtime boundaries.

---

## 2. Memory Layout & C-ABI Specification

### 2.1 C-Compatible Descriptor Structs (`dsp_abi.h`)

```cpp
#ifndef DSP_ABI_H
#define DSP_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Numerical Data Type Enums
typedef enum {
    DSP_FLOAT32   = 0,
    DSP_FLOAT64   = 1,
    DSP_COMPLEX64 = 2,  // std::complex<float> / np.complex64 / ComplexF32
    DSP_COMPLEX128= 3   // std::complex<double> / np.complex128 / ComplexF64
} dsp_datatype_t;

// C-ABI Descriptor for 1D Continuous Signal Buffers
typedef struct {
    void*           data;         // 64-byte aligned data pointer
    size_t          capacity;     // Total allocated elements
    size_t          size;         // Active valid elements
    double          sample_rate;  // Sampling frequency Fs (Hz)
    dsp_datatype_t  dtype;        // Element data type
    void*           owner_ref;    // Pointer to ownership token or ref-counter
    void (*free_cb)(void* owner); // Free callback for cross-language ownership release
} dsp_signal_buffer_abi_t;

// C-ABI Descriptor for N-Dimensional Tensors (Frames / Channels)
typedef struct {
    void*           data;         // Pointer to continuous underlying storage
    uint32_t        ndim;         // Number of dimensions
    int64_t*        shape;        // Size along each dimension
    int64_t*        strides;      // Element strides in memory
    dsp_datatype_t  dtype;        // Element data type
    void*           owner_ref;    // Ownership token
    void (*free_cb)(void* owner); // Free callback
} dsp_frame_tensor_abi_t;

#ifdef __cplusplus
}
#endif

#endif // DSP_ABI_H
```

---

## 3. Core C++ Implementation

The C++ core manages aligned memory allocation and exposes RAII wrappers.

```cpp
// SignalBuffer.hpp
#pragma once
#include "dsp_abi.h"
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace dsp::core {

template <typename T>
class SignalBuffer {
private:
    T* data_ptr_ = nullptr;
    size_t capacity_ = 0;
    size_t size_ = 0;
    double sample_rate_ = 1.0;

public:
    SignalBuffer(size_t capacity, double sample_rate = 1.0)
        : capacity_(capacity), size_(capacity), sample_rate_(sample_rate) {
        
        // 64-byte alignment for AVX-512 / SIMD operations
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(capacity * sizeof(T), 64);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, 64, capacity * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        data_ptr_ = static_cast<T*>(ptr);
    }

    ~SignalBuffer() {
        if (data_ptr_) {
#if defined(_MSC_VER)
            _aligned_free(data_ptr_);
#else
            free(data_ptr_);
#endif
        }
    }

    // Disable copy, enable move semantics
    SignalBuffer(const SignalBuffer&) = delete;
    SignalBuffer& operator=(const SignalBuffer&) = delete;
    SignalBuffer(SignalBuffer&& other) noexcept 
        : data_ptr_(other.data_ptr_), capacity_(other.capacity_), 
          size_(other.size_), sample_rate_(other.sample_rate_) {
        other.data_ptr_ = nullptr;
    }

    T* data() { return data_ptr_; }
    const T* data() const { return data_ptr_; }
    size_t size() const { return size_; }
    double sample_rate() const { return sample_rate_; }

    // Export to C-ABI struct for Zero-Copy Interop
    dsp_signal_buffer_abi_t export_abi() {
        dsp_signal_buffer_abi_t abi{};
        abi.data = static_cast<void*>(data_ptr_);
        abi.capacity = capacity_;
        abi.size = size_;
        abi.sample_rate = sample_rate_;
        abi.dtype = std::is_same_v<T, std::complex<float>> ? DSP_COMPLEX64 : DSP_FLOAT32;
        abi.owner_ref = this; // Self ownership reference
        abi.free_cb = [](void* owner) {
            // Callback invoked when foreign language runtime releases reference
            // Managed via shared_ptr wrappers in production
        };
        return abi;
    }
};

} // namespace dsp::core
```

---

## 4. Language-Specific Zero-Copy Integrations

### 4.1 Python (NumPy Buffer Protocol via `pybind11`)

By exposing the Python **Buffer Protocol** or returning `py::array_t` with pointing buffers, Python accesses C++ allocated memory directly without copying.

```cpp
// pybind_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "SignalBuffer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dsp_core_py, m) {
    py::class_<dsp::core::SignalBuffer<std::complex<float>>, 
               std::shared_ptr<dsp::core::SignalBuffer<std::complex<float>>>>(m, "SignalBuffer")
        .def(py::init<size_t, double>(), py::arg("capacity"), py::arg("sample_rate") = 1.0)
        .def_property_readonly("size", &dsp::core::SignalBuffer<std::complex<float>>::size)
        .def_property_readonly("sample_rate", &dsp::core::SignalBuffer<std::complex<float>>::sample_rate)
        
        // Zero-copy NumPy view export
        .def("to_numpy", [](std::shared_ptr<dsp::core::SignalBuffer<std::complex<float>>>& self) {
            return py::array_t<std::complex<float>>(
                { static_cast<ssize_t>(self->size()) },                      // Shape
                { sizeof(std::complex<float>) },                             // Strides
                self->data(),                                                // Pointer
                py::cast(self)                                               // Owner (keeps C++ object alive)
            );
        });
}
```

**Python Usage:**
```python
import dsp_core_py as dsp
import numpy as np

# Allocate in C++ core
buf = dsp.SignalBuffer(1024, 1e6)

# Get zero-copy NumPy array view
np_view = buf.to_numpy()

# In-place manipulation directly modifies C++ memory
np_view[:10] = 1.0 + 2.0j
print(f"NumPy array shape: {np_view.shape}, dtype: {np_view.dtype}")
```

---

### 4.2 Julia Interoperability (`unsafe_wrap`)

Julia can wrap raw pointer structures without memory duplication using `unsafe_wrap`.

```julia
# SignalBuffer.jl
module JuliaDSP

using Libdl

# Map C-ABI struct to Julia
struct DSPSignalBufferABI
    data::Ptr{ComplexF32}
    capacity::Csize_t
    size::Csize_t
    sample_rate::Cdouble
    dtype::Cint
    owner_ref::Ptr{Cvoid}
    free_cb::Ptr{Cvoid}
end

"""
    wrap_signal_buffer(abi::DSPSignalBufferABI)

Wraps a `DSPSignalBufferABI` pointer into a zero-copy Julia 1D Vector.
"""
function wrap_signal_buffer(abi::DSPSignalBufferABI)
    # Zero-copy wrap: Julia array directly points to C++ aligned memory
    # own=false ensures Julia GC does NOT free C++ managed buffer
    julia_vec = unsafe_wrap(Vector{ComplexF32}, abi.data, abi.size; own=false)
    return julia_vec
end

end # module
```

---

## 5. Memory Ownership & Safety Rules

| Runtime | Allocation Source | Memory Deallocation Responsibility | Zero-Copy Mechanism |
| :--- | :--- | :--- | :--- |
| **C++ Core** | `posix_memalign` / `_aligned_malloc` | C++ `std::shared_ptr` / RAII | Direct pointer usage |
| **Python** | Wrapped from C++ | Handled by Python `py::cast(owner)` ref-counting | NumPy Buffer Protocol (`py::array_t`) |
| **Julia** | Wrapped from C++ | C++ retain/release callbacks | `unsafe_wrap(Array, ptr, dims; own=false)` |

1. **Lifetime Guarantee:** Foreign runtime object wrappers MUST retain a reference count to the underlying C++ object instance until foreign garbage collection cleans them up.
2. **Alignment Safety:** All pointers must guarantee **64-byte alignment** to support AVX-512 SIMD vectorization routines without memory misalignment faults.
3. **Thread Safety:** Multiple concurrent reads across Python threads or Julia tasks are safe. Concurrent writes must be guarded via synchronization locks at the pipeline stage level.