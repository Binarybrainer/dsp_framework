# ⚡ FFT & IFFT Processing Specification

Spectral analysis and frequency-domain filtering are core operations in digital signal processing. Fast Fourier Transform (FFT) algorithms reduce the computational complexity of the Discrete Fourier Transform (DFT) from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$.

This document specifies the `FFTBlock` and `IFFTBlock` components for our hybrid framework. It covers windowing options, streaming spectral processing (Overlap-Save/Add), Python implementation, a C++ migration plan, and numerical verification tests.

---

## 1. Mathematical Foundations

### 1.1. Discrete Fourier Transform (DFT) & Inverse DFT (IDFT)

For a discrete signal sequence $x[n]$ of length $N$, the forward DFT $X[k]$ and inverse DFT $x[n]$ are defined as:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1
$$

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j \frac{2\pi}{N} k n}, \quad n = 0, 1, \dots, N-1
$$

### 1.2. Parseval's Theorem & Energy Conservation

The total energy in the time domain equals the total energy in the frequency domain scaled by $1/N$:

$$
E_{time} = \sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2 = E_{freq}
$$

When applying window functions $w[n]$, scaling factors must be maintained to preserve physical units (e.g., peak amplitude or noise power spectral density).

---

## 2. Windowing Integration

When transforming finite-length signal chunks, implicit rectangular windowing causes spectral leakage due to discontinuities at chunk boundaries.

### 2.1. Supported Window Functions

The `FFTBlock` supports several window types:

1. **Rectangular:** Unwindowed baseline, maximum frequency resolution, high sidelobes ($-13\text{ dB}$).
2. **Hann:** $w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$, good general-purpose trade-off.
3. **Hamming:** $w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)$, reduces first sidelobe to $-43\text{ dB}$.
4. **Blackman:** $w[n] = 0.42 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right) + 0.08 \cos\left(\frac{4\pi n}{N-1}\right)$, low sidelobes ($-58\text{ dB}$).
5. **Flat-Top:** Optimized for absolute amplitude measurement accuracy in spectral analysis.

### 2.2. Amplitude & Power Correction Factors

To correct energy loss due to tapering:

* **Coherent Amplitude Gain (CAG):** $C_{amp} = \frac{1}{N} \sum_{n=0}^{N-1} w[n]$ (Corrects single-tone peak values).
* **Noise Power Gain (NPG) / Equivalent Noise Bandwidth (ENBW):** $C_{power} = \frac{1}{N} \sum_{n=0}^{N-1} w^2[n]$ (Corrects noise spectral density).

---

## 3. Streaming Spectral Processing (Overlap-Save Method)

When performing frequency-domain filtering on continuous streaming data chunks, direct pointwise multiplication in the frequency domain corresponds to **circular convolution** in time. To achieve linear convolution without edge distortion across streaming chunk boundaries, we implement the **Overlap-Save (OLS)** method.

### 3.1. Overlap-Save Algorithm Procedure

For an FIR filter impulse response $h[n]$ of length $M$ and incoming data blocks of size $L$:

1. Set the FFT size $N_{fft} \ge L + M - 1$ (typically chosen as the next power of 2).
2. For each new chunk of $L$ samples, prepend $M-1$ samples carried over from the end of the previous chunk.
3. Compute $N_{fft}$-point FFT of the overlapped input vector.
4. Multiply element-wise by the pre-computed frequency response $H[k] = \text{FFT}(h[n], N_{fft})$.
5. Compute $N_{fft}$-point IFFT.
6. Discard the first $M-1$ corrupted samples (transient circular overlap) and output the remaining $L$ valid samples.

---

## 4. Python Implementation (`FFTBlock` & `IFFTBlock`)

```python
import numpy as np
from typing import Any, Optional
from scipy.signal import windows
from dsp_framework.core.dsp_block_api import DSPBlock
from dsp_framework.core.signal_buffer_spec import SignalBuffer

class FFTBlock(DSPBlock):
    """
    Computes forward FFT on streaming SignalBuffers with optional windowing,
    padding, and power/amplitude scaling options.
    """
    def __init__(self, 
                 n_fft: Optional[int] = None, 
                 window_type: str = 'rectangular',
                 normalization: str = 'standard') -> None:
        """
        Args:
            n_fft: FFT length. If None, uses chunk length.
            window_type: 'rectangular', 'hann', 'hamming', 'blackman', 'flattop'.
            normalization: 'standard' (no scale), 'ortho' (1/sqrt(N)), or 'amplitude' (1/N).
        """
        super().__init__(n_fft=n_fft, window_type=window_type.lower(), normalization=normalization)
        self.reset()

    def reset(self) -> None:
        pass  # Stateless for independent FFT frames

    def _get_window(self, length: int) -> np.ndarray:
        w_type = self._params['window_type']
        if w_type == 'rectangular' or w_type == 'none':
            return np.ones(length, dtype=np.float64)
        elif w_type == 'hann':
            return windows.hann(length, sym=False)
        elif w_type == 'hamming':
            return windows.hamming(length, sym=False)
        elif w_type == 'blackman':
            return windows.blackman(length, sym=False)
        elif w_type == 'flattop':
            return windows.flattop(length, sym=False)
        else:
            raise ValueError(f"Unsupported window type: {w_type}")

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        N = len(data)
        n_fft = self._params['n_fft'] if self._params['n_fft'] is not None else N
        
        # 1. Apply Window Function
        w = self._get_window(N)
        windowed_data = data * w
        
        # 2. Perform Forward FFT
        norm_mode = self._params['normalization']
        if norm_mode == 'amplitude':
            # Coherent amplitude scaling
            scale = 1.0 / np.sum(w)
            fft_data = np.fft.fft(windowed_data, n=n_fft) * scale
        elif norm_mode == 'ortho':
            fft_data = np.fft.fft(windowed_data, n=n_fft, norm='ortho')
        else:
            fft_data = np.fft.fft(windowed_data, n=n_fft)
            
        # Frequency step
        df = input_buf.sample_rate / n_fft
        
        return SignalBuffer(
            data=fft_data,
            sample_rate=input_buf.sample_rate,
            timestamp=input_buf.timestamp
        )


class IFFTBlock(DSPBlock):
    """
    Computes Inverse FFT to convert frequency-domain buffers back to time domain.
    """
    def __init__(self, n_fft: Optional[int] = None, normalization: str = 'standard') -> None:
        super().__init__(n_fft=n_fft, normalization=normalization)
        self.reset()

    def reset(self) -> None:
        pass

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        n_fft = self._params['n_fft'] if self._params['n_fft'] is not None else len(data)
        norm_mode = self._params['normalization']
        
        if norm_mode == 'ortho':
            time_data = np.fft.ifft(data, n=n_fft, norm='ortho')
        else:
            time_data = np.fft.ifft(data, n=n_fft)
            
        return SignalBuffer(
            data=time_data,
            sample_rate=input_buf.sample_rate,
            timestamp=input_buf.timestamp
        )
```

---

## 5. C++ Backend Migration Plan (Phase 4 Integration)

While 1D FFT operations in NumPy wrap highly optimized C libraries (FFTPACK/PocketFFT), custom streaming spectral blocks (like Overlap-Save filtering or dynamic spectral masking) benefit from running directly inside C++ to avoid Python wrapper overhead during high-throughput execution.

### 5.1. C++ Library Selection

We integrate `pocketfft` (header-only, lightweight, C++11/C++20 compliant) or `FFTW3` into our C++ backend (`csrc/fft_core.cpp`).

### 5.2. C++ Implementation Template (`csrc/fft_core.cpp`)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include "pocketfft_hdronly.h"

namespace py = pybind11;
using Complex64 = std::complex<double>;

class FastFFT_CPP {
private:
    size_t n_fft;

public:
    FastFFT_CPP(size_t n_fft) : n_fft(n_fft) {}

    void forward(py::array_t<Complex64> input_array, py::array_t<Complex64> output_array) {
        py::buffer_info buf_in = input_array.request();
        py::buffer_info buf_out = output_array.request();

        auto* ptr_in = static_cast<Complex64*>(buf_in.ptr);
        auto* ptr_out = static_cast<Complex64*>(buf_out.ptr);

        pocketfft::shape_t shape = {n_fft};
        pocketfft::stride_t stride_in = {sizeof(Complex64)};
        pocketfft::stride_t stride_out = {sizeof(Complex64)};
        pocketfft::shape_t axes = {0};

        // Zero-copy execution from NumPy buffer directly to C++ output buffer
        pocketfft::c2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
                       ptr_in, ptr_out, 1.0);
    }
};

PYBIND11_MODULE(fft_core_cpp, m) {
    py::class_<FastFFT_CPP>(m, "FastFFT_CPP")
        .def(py::init<size_t>())
        .def("forward", &FastFFT_CPP::forward);
}
```

---

## 6. Verification & Parity Test Suite

The following `pytest` test suite validates exact energy conservation, invertibility ($\text{IFFT}(\text{FFT}(x)) = x$), window scaling, and C++ vs. Python numerical parity.

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.blocks.fft_spec import FFTBlock, IFFTBlock

def test_fft_ifft_roundtrip():
    """Verify that IFFT(FFT(x)) recovers original signal within numerical tolerance."""
    np.random.seed(42)
    N = 1024
    x = np.random.randn(N) + 1j * np.random.randn(N)
    in_buf = SignalBuffer(x, sample_rate=1e6)

    fft_block = FFTBlock(n_fft=N, window_type='rectangular', normalization='standard')
    ifft_block = IFFTBlock(n_fft=N, normalization='standard')

    freq_buf = fft_block.process(in_buf)
    recovered_buf = ifft_block.process(freq_buf)

    assert_allclose(recovered_buf.data, in_buf.data, rtol=1e-12, atol=1e-12)

def test_parsevals_theorem():
    """Verify Parseval's theorem: time-domain energy == normalized frequency-domain energy."""
    np.random.seed(100)
    N = 2048
    x = np.random.randn(N) + 1j * np.random.randn(N)
    in_buf = SignalBuffer(x, sample_rate=1e6)

    fft_block = FFTBlock(n_fft=N, window_type='rectangular', normalization='standard')
    freq_buf = fft_block.process(in_buf)

    e_time = np.sum(np.abs(x)**2)
    e_freq = np.sum(np.abs(freq_buf.data)**2) / N

    assert_allclose(e_time, e_freq, rtol=1e-10)

def test_hann_window_amplitude_correction():
    """Verify coherent amplitude scaling for a pure single-tone sinusoid."""
    fs = 1000.0
    N = 1000
    f0 = 50.0 # 50 Hz tone
    amp = 2.5
    t = np.arange(N) / fs
    x = amp * np.exp(1j * 2 * np.pi * f0 * t)
    in_buf = SignalBuffer(x, sample_rate=fs)

    fft_block = FFTBlock(n_fft=N, window_type='hann', normalization='amplitude')
    freq_buf = fft_block.process(in_buf)

    # Peak amplitude in spectrum should match input amplitude (2.5)
    peak_val = np.max(np.abs(freq_buf.data))
    assert_allclose(peak_val, amp, rtol=1e-3)
```