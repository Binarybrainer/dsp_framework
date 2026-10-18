# 🏎️ Phase 4: Profiling & Backend Migration Specification

This phase executes the core of our "Hybrid Performance Strategy." We transition from a purely mathematical Python prototype to a high-performance system by systematically measuring code execution, writing targeted C++20 extensions, and integrating them seamlessly using zero-copy memory via `pybind11`.

## 1. Performance Profiling

Before writing any C++, we must prove where the bottlenecks exist. Premature optimization wastes time.

### 1.1. High-Level Profiling with `cProfile`
We use Python's built-in `cProfile` to analyze the entire `DSPPipeline` execution time. This tells us *which* block is the slowest.

```python
import cProfile
import pstats

def run_simulation():
    # Assume a pipeline with AWGN, Fading, and PLL
    pipeline.process(large_signal_buffer)

# Run profiler
profiler = cProfile.Profile()
profiler.enable()
run_simulation()
profiler.disable()

# Print top 10 slowest functions
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(10)
```

### 1.2. Micro-Profiling with `line_profiler`
Once `cProfile` identifies a slow block (e.g., `PLLBlock.process`), we use `line_profiler` to see exactly which line of the mathematical loop is consuming the CPU.

```python
# Install: pip install line_profiler
from line_profiler import profile

class PLLBlock(DSPBlock):
    @profile  # Decorate the bottleneck function
    def process(self, input_buf: SignalBuffer, **kwargs) -> SignalBuffer:
        # line_profiler will output the exact time spent on each line
        # of the sample-by-sample for-loop here.
        ...
```

## 2. C++ Backend Infrastructure & Zero-Copy Memory

To ensure C++ actually speeds up our code, we must avoid copying large data arrays between Python and C++ memory spaces.

### 2.1. CMake & pybind11 Setup
Our `CMakeLists.txt` and `pyproject.toml` (created in Phase 1) handle the build system. `scikit-build-core` will compile the C++ code into a `.so` (Linux/Mac) or `.pyd` (Windows) library file that Python can `import` directly.

### 2.2. The Zero-Copy Interface
We use `pybind11::array_t` to pass the NumPy array directly to C++. We extract the raw memory pointer and array length.

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>

namespace py = pybind11;

// The C++ processing function
void process_pll_cpp(
    py::array_t<std::complex<float>> input_array,
    py::array_t<std::complex<float>> output_array,
    float kp, float ki) 
{
    // Request buffer information (Zero-copy memory mapping)
    py::buffer_info buf_in = input_array.request();
    py::buffer_info buf_out = output_array.request();

    // Extract raw C++ pointers
    auto* ptr_in = static_cast<std::complex<float>*>(buf_in.ptr);
    auto* ptr_out = static_cast<std::complex<float>*>(buf_out.ptr);
    size_t num_samples = buf_in.shape[0];

    // High-performance C++ loop
    for (size_t i = 0; i < num_samples; i++) {
        // PLL logic here...
        ptr_out[i] = ptr_in[i] * /* phase rotation */;
    }
}

// Bind to Python
PYBIND11_MODULE(dsp_core_cpp, m) {
    m.def("process_pll", &process_pll_cpp, "Fast PLL processing");
}
```

## 3. Targeted Migration & API Wrapping

Once the C++ module (`dsp_core_cpp`) is compiled, we wrap it in a standard Python `DSPBlock` so the rest of the framework doesn't know it's interacting with C++.

```python
import numpy as np
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

# Import our compiled C++ module
import dsp_core_cpp

class FastPLLBlock(DSPBlock):
    """
    High-performance PLL block utilizing the C++ backend.
    """
    def __init__(self, kp: float = 0.1, ki: float = 0.01):
        super().__init__(kp=kp, ki=ki)
        self.reset()
        
    def reset(self):
        # Reset internal C++ state if managed via a class, 
        # or reset Python wrapper states.
        pass

    def process(self, input_buf: SignalBuffer, **kwargs) -> SignalBuffer:
        # Pre-allocate output array in Python to manage memory safely
        out_data = np.zeros_like(input_buf.data)
        
        # Call the zero-copy C++ function
        dsp_core_cpp.process_pll(
            input_buf.data, 
            out_data, 
            self._params['kp'], 
            self._params['ki']
        )
        
        return input_buf.clone_metadata(out_data)
```

## 4. Numerical Parity Testing

Before replacing the pure Python prototype `PLLBlock` with `FastPLLBlock`, we must mathematically guarantee they are identical. We use `pytest` and `numpy.testing`.

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

def test_pll_c_cpp_parity():
    # 1. Generate test signal
    test_data = np.random.randn(10000) + 1j * np.random.randn(10000)
    input_buf = SignalBuffer(test_data, sample_rate=1e6)
    
    # 2. Instantiate both blocks
    py_pll = PLLBlock(kp=0.1, ki=0.01)
    cpp_pll = FastPLLBlock(kp=0.1, ki=0.01)
    
    # 3. Process
    out_py = py_pll.process(input_buf)
    out_cpp = cpp_pll.process(input_buf)
    
    # 4. Assert mathematical equivalence
    # rtol: relative tolerance, atol: absolute tolerance 
    # (accounts for minor floating-point differences between Python and C++)
    assert_allclose(out_py.data, out_cpp.data, rtol=1e-5, atol=1e-8)
```

By passing this test, you guarantee that moving to C++ introduces zero logic regressions while significantly reducing execution time.