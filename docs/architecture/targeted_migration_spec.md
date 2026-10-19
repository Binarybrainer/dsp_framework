# 🎯 Targeted Migration: C++ Implementation & Parity Testing

Once `cProfile` and `line_profiler` have identified a bottleneck (typically a stateful loop like a PLL, CDR, or Adaptive Equalizer), we execute the **Targeted Migration**. 

Instead of rewriting the entire framework, we replace *only* the internal loop of the specific bottleneck block with C++20, wrap it via `pybind11`, and seamlessly plug it back into the Python pipeline. Crucially, we enforce strict numerical parity testing to ensure no logic regressions occur.

## 1. C++20 Implementation of a Stateful Block

Unlike simple vectorized functions, blocks like the PLL must retain state (e.g., phase memory) across multiple `process()` calls. We implement this as a C++ class.

### `csrc/pll_core.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>
#include <numbers> // C++20 math constants

namespace py = pybind11;
using Complex64 = std::complex<double>;

class PLL_CPP {
private:
    double kp;
    double ki;
    double center_freq;
    double phase_state;
    double integral_state;

public:
    PLL_CPP(double kp, double ki, double center_freq) 
        : kp(kp), ki(ki), center_freq(center_freq), phase_state(0.0), integral_state(0.0) {}

    void reset() {
        phase_state = 0.0;
        integral_state = 0.0;
    }

    void process(py::array_t<Complex64> input_array, py::array_t<Complex64> output_array) {
        py::buffer_info buf_in = input_array.request();
        py::buffer_info buf_out = output_array.request();

        auto* ptr_in = static_cast<Complex64*>(buf_in.ptr);
        auto* ptr_out = static_cast<Complex64*>(buf_out.ptr);
        size_t num_samples = buf_in.shape[0];

        for (size_t i = 0; i < num_samples; ++i) {
            // NCO Output
            Complex64 nco_out = std::polar(1.0, phase_state);
            
            // Phase Error: Im(data * conj(nco_out))
            double phase_error = std::imag(ptr_in[i] * std::conj(nco_out));
            
            // PI Loop Filter
            integral_state += ki * phase_error;
            double control_voltage = kp * phase_error + integral_state;
            
            // Update NCO Phase
            phase_state += center_freq + control_voltage;
            
            // Wrap phase to [-pi, pi]
            constexpr double PI = std::numbers::pi;
            phase_state = std::fmod(phase_state + PI, 2.0 * PI);
            if (phase_state < 0) phase_state += 2.0 * PI;
            phase_state -= PI;
            
            // Output de-rotated signal
            ptr_out[i] = ptr_in[i] * std::conj(nco_out);
        }
    }
};

// pybind11 Class Wrapper
PYBIND11_MODULE(dsp_core_cpp, m) {
    py::class_<PLL_CPP>(m, "PLL_CPP")
        .def(py::init<double, double, double>())
        .def("reset", &PLL_CPP::reset)
        .def("process", &PLL_CPP::process);
}
```

## 2. Python `DSPBlock` API Wrapper

We now create a Python class that inherits from `DSPBlock` to maintain complete API compatibility with the rest of the pipeline, but internally delegates the work to the C++ instance.

### `src/dsp_framework/blocks/fast_pll.py`

```python
import numpy as np
from typing import Any
from dsp_framework.core.dsp_block_api import DSPBlock
from dsp_framework.core.signal_buffer_spec import SignalBuffer
import dsp_core_cpp  # The compiled C++ module

class FastPLLBlock(DSPBlock):
    """
    High-performance C++ backend PLL perfectly matching the Python DSPBlock API.
    """
    def __init__(self, kp: float = 0.1, ki: float = 0.01, center_freq: float = 0.0) -> None:
        super().__init__(kp=kp, ki=ki, center_freq=center_freq)
        # Instantiate the C++ backend object
        self._cpp_backend = dsp_core_cpp.PLL_CPP(kp, ki, center_freq)
        
    def reset(self) -> None:
        self._cpp_backend.reset()

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        # Ensure input is contiguous double-precision complex
        data_in = np.ascontiguousarray(input_buf.data, dtype=np.complex128)
        data_out = np.zeros_like(data_in)
        
        # Call the zero-copy C++ processing loop
        self._cpp_backend.process(data_in, data_out)
        
        return input_buf.clone_metadata(data_out)
```

## 3. Numerical Parity Testing

Before we fully replace the old Python block, we must write a `pytest` script to guarantee mathematical equivalence. Floating-point math can sometimes differ slightly between Python and C++ due to compiler optimizations or different mathematical libraries.

### `tests/test_pll_parity.py`

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.blocks.pll_specification import PLLBlock       # Pure Python
from dsp_framework.blocks.fast_pll import FastPLLBlock            # C++ Backend

def test_pll_numerical_parity():
    # 1. Generate a test signal (e.g., QPSK with Phase Offset)
    np.random.seed(42)
    N = 50000
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], N) / np.sqrt(2)
    
    # Add a deterministic frequency offset
    t = np.arange(N)
    freq_offset = 1e-4
    rx_signal = symbols * np.exp(1j * 2 * np.pi * freq_offset * t)
    
    # Wrap in SignalBuffer
    test_buf = SignalBuffer(rx_signal, sample_rate=1.0)
    
    # 2. Instantiate both blocks with identical parameters
    kp, ki = 0.05, 0.001
    py_pll = PLLBlock(kp=kp, ki=ki)
    cpp_pll = FastPLLBlock(kp=kp, ki=ki)
    
    # 3. Process the data
    py_out = py_pll.process(test_buf)
    cpp_out = cpp_pll.process(test_buf)
    
    # 4. Assert Equivalence
    # We use assert_allclose to account for minor C++ vs Python float rounding differences
    assert_allclose(
        actual=cpp_out.data, 
        desired=py_out.data, 
        rtol=1e-5,  # Relative tolerance
        atol=1e-8,  # Absolute tolerance
        err_msg="C++ PLL output diverged from Python PLL prototype!"
    )
```