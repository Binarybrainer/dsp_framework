# 🧩 Developer Guide: Guidelines for Adding New DSP Blocks

This document provides a step-by-step walkthrough for adding a new DSP module (e.g., a **Gain Block**, **CDR Filter**, or **PLL** component) to the framework. 

To ensure full multi-language compatibility, every new module must be implemented in the C++ core, bound to Python and Julia, and verified with unit tests.

---

## 🛠️ Summary Workflow

When introducing a new DSP module `<module_name>`, follow these 5 steps:

1. **C++ Core Header:** Define module interface in `include/dsp/blocks/<module_name>.hpp`.
2. **C++ Implementation:** Implement processing algorithms in `src/blocks/<module_name>.cpp`.
3. **Python Binding:** Register class/methods in `src/bindings/python/bind_blocks.cpp`.
4. **Julia Binding:** Expose C-ABI or CxxWrap interfaces in `julia/src/abi_bindings.jl`.
5. **Unit Tests:** Add test suites in `tests/cpp/test_<module_name>.cpp` and `tests/python/test_<module_name>.py`.

---

## Step 1: C++ Core Header (`include/dsp/blocks/gain_block.hpp`)

Define the class inheriting from `DSPModule` or defining a standalone interface using standard modern C++20 conventions.

```cpp
#pragma once

#include "dsp/core/DSPModule.hpp"
#include "dsp/core/SignalBuffer.hpp"
#include <complex>

namespace dsp::blocks {

class GainBlock : public core::DSPModule {
private:
    float gain_factor_;

public:
    explicit GainBlock(float gain_factor = 1.0f);
    ~GainBlock() override = default;

    // Setters and Getters
    void set_gain(float gain) { gain_factor_ = gain; }
    float get_gain() const { return gain_factor_; }

    // Core processing routine
    void process(const std::complex<float>* input, 
                 std::complex<float>* output, 
                 size_t num_samples);

    // DSPModule Interface Overrides
    void reset() override;
};

} // namespace dsp::blocks
```

---

## Step 2: C++ Implementation (`src/blocks/gain_block.cpp`)

Implement the core signal processing algorithms. Optimize loops for SIMD vectorization where applicable.

```cpp
#include "dsp/blocks/gain_block.hpp"

namespace dsp::blocks {

GainBlock::GainBlock(float gain_factor) 
    : gain_factor_(gain_factor) {}

void GainBlock::process(const std::complex<float>* input, 
                        std::complex<float>* output, 
                        size_t num_samples) {
    // Vectorizable sample processing
    for (size_t i = 0; i < num_samples; ++i) {
        output[i] = input[i] * gain_factor_;
    }
}

void GainBlock::reset() {
    // Reset internal state/memory if applicable
}

} // namespace dsp::blocks
```

---

## Step 3: Python Binding (`src/bindings/python/bind_blocks.cpp`)

Register the C++ class with `pybind11` so Python users can instantiate and manipulate the block.

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "dsp/blocks/gain_block.hpp"

namespace py = pybind11;
using namespace dsp::blocks;

void bind_gain_block(py::module_& m) {
    py::class_<GainBlock, std::shared_ptr<GainBlock>>(m, "GainBlock")
        .def(py::init<float>(), py::arg("gain_factor") = 1.0f)
        .def_property("gain", &GainBlock::get_gain, &GainBlock::set_gain)
        .def("process", [](GainBlock& self, 
                           py::array_t<std::complex<float>> input, 
                           py::array_t<std::complex<float>> output) {
            auto buf_in = input.request();
            auto buf_out = output.request();

            if (buf_in.size != buf_out.size) {
                throw std::runtime_error("Input and output buffer sizes must match.");
            }

            self.process(
                static_cast<const std::complex<float>*>(buf_in.ptr),
                static_cast<std::complex<float>*>(buf_out.ptr),
                buf_in.size
            );
        }, "Process signal samples with gain block");
}
```

---

## Step 4: Julia Binding (`julia/src/abi_bindings.jl`)

Expose the function via C-ABI or CxxWrap for zero-copy access in Julia.

```julia
# Expose GainBlock via C-ABI wrapper functions in julia/src/abi_bindings.jl

"""
    GainBlock(gain_factor::Float32)

Julia wrapper for C++ GainBlock instance.
"""
mutable struct GainBlock
    cpp_handle::Ptr{Cvoid}

    function GainBlock(gain_factor::Float32 = 1.0f0)
        handle = ccall((:dsp_gain_block_create, LIB_DSP_CORE), Ptr{Cvoid}, (Cfloat,), gain_factor)
        obj = new(handle)
        finalizer(obj) do x
            ccall((:dsp_gain_block_destroy, LIB_DSP_CORE), Cvoid, (Ptr{Cvoid},), x.cpp_handle)
        end
        return obj
    end
end

function process!(block::GainBlock, input::Vector{ComplexF32}, output::Vector{ComplexF32})
    @assert length(input) == length(output) "Input and output lengths must match"
    ccall(
        (:dsp_gain_block_process, LIB_DSP_CORE), 
        Cvoid, 
        (Ptr{Cvoid}, Ptr{ComplexF32}, Ptr{ComplexF32}, Csize_t),
        block.cpp_handle, input, output, length(input)
    )
end
```

---

## Step 5: Unit Tests

### 5.1 C++ Unit Test (`tests/cpp/test_gain_block.cpp`)

```cpp
#include <gtest/gtest.h>
#include "dsp/blocks/gain_block.hpp"
#include <vector>

TEST(GainBlockTest, ScaledOutputTest) {
    dsp::blocks::GainBlock gain_block(2.0f);

    std::vector<std::complex<float>> input = {{1.0f, 2.0f}, {3.0f, -1.0f}};
    std::vector<std::complex<float>> output(2);

    gain_block.process(input.data(), output.data(), input.size());

    EXPECT_EQ(output[0], std::complex<float>(2.0f, 4.0f));
    EXPECT_EQ(output[1], std::complex<float>(6.0f, -2.0f));
}
```

### 5.2 Python Unit Test (`tests/python/test_gain_block.py`)

```python
import pytest
import numpy as np
import dsp_framework_py as dsp

def test_gain_block_scaling():
    gain_block = dsp.GainBlock(3.0)
    assert gain_block.gain == 3.0

    in_signal = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64)
    out_signal = np.zeros_like(in_signal)

    gain_block.process(in_signal, out_signal)

    expected = in_signal * 3.0
    np.testing.assert_allclose(out_signal, expected, rtol=1e-6)
```

---

## 📋 Checklist for Verification

Before opening a Pull Request with a new DSP block, ensure:

- [ ] C++ Header & Implementation compiled without warnings (`-Wall -Wextra`).
- [ ] Python bindings registered and imported cleanly in `pytest`.
- [ ] Julia CFFI functions mapped without memory leaks.
- [ ] All C++ and Python unit tests pass (`ctest` & `pytest`).
- [ ] Code is formatted according to project conventions (`clang-format`).