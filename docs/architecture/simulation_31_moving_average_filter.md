# 🧹 Simulation 31: Design and Apply a Moving Average Filter

The Moving Average (MA) filter is one of the simplest and most widely used digital filters. In the time domain, it acts as an optimal unweighted smoother for reducing random white noise while keeping a sharp step response. 

This document details the mathematical properties of the Moving Average filter, its Python implementation as a stateful `DSPBlock`, and a complete simulation testbench.

---

## 1. Mathematical Background

### 1.1. Time-Domain Difference Equation

An $M$-point moving average filter takes the arithmetic mean of the current sample and $M-1$ previous samples:

$$
y[n] = \frac{1}{M} \sum_{k=0}^{M-1} x[n-k]
$$

This is an FIR filter with $M$ equal coefficients (taps):

$$
b_k = \frac{1}{M}, \quad \text{for } k = 0, 1, \dots, M-1
$$

### 1.2. Frequency-Domain Response

Taking the Z-transform yields the transfer function $H(z)$:

$$
H(z) = \frac{1}{M} \sum_{k=0}^{M-1} z^{-k} = \frac{1}{M} \frac{1 - z^{-M}}{1 - z^{-1}}
$$

Substituting $z = e^{j\omega}$ gives the discrete-time frequency response:

$$
H(e^{j\omega}) = \frac{1}{M} e^{-j\omega \frac{M-1}{2}} \frac{\sin(\omega M / 2)}{\sin(\omega / 2)}
$$

* **Magnitude Response:** Follows a Dirichlet / Sinc-like shape. Nulls occur at normalized frequencies $\omega = \frac{2\pi k}{M}$ for $k = 1, 2, \dots, M-1$.
* **Linear Phase:** Because the filter taps are symmetric ($b_k = b_{M-1-k}$), the phase delay is constant:

$$
\tau_g = \frac{M - 1}{2} \text{ samples}
$$

### 1.3. Noise Reduction Property

If the input $x[n] = s[n] + w[n]$ consists of a signal $s[n]$ contaminated by zero-mean uncorrelated additive white noise $w[n]$ with variance $\sigma_{in}^2$, the output noise variance $\sigma_{out}^2$ is reduced by a factor of $M$:

$$
\sigma_{out}^2 = \frac{\sigma_{in}^2}{M} \implies \sigma_{out} = \frac{\sigma_{in}}{\sqrt{M}}
$$

---

## 2. Python Implementation (`MovingAverageFilterBlock`)

To maintain exact continuity across streaming buffer chunks, the implementation uses `scipy.signal.lfilter` with filter state (`zi`) tracking.

```python
import numpy as np
from scipy.signal import lfilter, lfilter_zi
from typing import Any
from dsp_framework.core.dsp_block_api import DSPBlock
from dsp_framework.core.signal_buffer_spec import SignalBuffer

class MovingAverageFilterBlock(DSPBlock):
    """
    Stateful M-point Moving Average (Boxcar) FIR Filter Block.
    Smooths input signals while preserving state across chunks.
    """
    def __init__(self, window_size: int = 5) -> None:
        """
        Args:
            window_size: Length of the moving average window (M).
        """
        if window_size < 1:
            raise ValueError("Window size M must be an integer >= 1.")
            
        taps = np.ones(window_size, dtype=np.float64) / window_size
        super().__init__(window_size=window_size, taps=taps)
        self.reset()

    def reset(self) -> None:
        """Reset internal filter memory for streaming continuity."""
        taps = self._params['taps']
        # Initialize filter state delay line to zeros
        self.zi = lfilter_zi(b=taps, a=[1.0]) * 0.0

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        taps = self._params['taps']
        
        # Apply FIR filtering with state tracking
        if np.iscomplexobj(data):
            # Process Real and Imaginary parts separately if complex
            y_real, zi_real = lfilter(taps, [1.0], data.real, zi=self.zi.real)
            y_imag, zi_imag = lfilter(taps, [1.0], data.imag, zi=self.zi.imag)
            y = y_real + 1j * y_imag
            self.zi = zi_real + 1j * zi_imag
        else:
            y, self.zi = lfilter(taps, [1.0], data, zi=self.zi)

        return input_buf.clone_metadata(y)
```

---

## 3. Simulation Testbench & Verification Script

The following script evaluates a 10-point moving average filter on a step function contaminated with Gaussian white noise, verifying both time-domain smoothing and statistical noise reduction.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.blocks.moving_average import MovingAverageFilterBlock

def run_simulation_31():
    # 1. Simulation Parameters
    fs = 1e3            # 1 kHz sample rate
    N = 500             # 500 samples (0.5 seconds)
    M = 10              # 10-point moving average window
    sigma_noise = 0.5   # Input noise standard deviation
    
    # 2. Construct Clean Signal (Step Signal + Sine wave)
    t = np.arange(N) / fs
    clean_signal = np.where(t >= 0.1, 1.0, 0.0) + 0.3 * np.sin(2 * np.pi * 10 * t)
    
    # 3. Add Additive White Gaussian Noise
    np.random.seed(42)
    noise = np.random.normal(0, sigma_noise, N)
    noisy_signal = clean_signal + noise
    
    input_buf = SignalBuffer(data=noisy_signal, sample_rate=fs)
    
    # 4. Process Through Moving Average Block
    ma_filter = MovingAverageFilterBlock(window_size=M)
    output_buf = ma_filter.process(input_buf)
    
    # 5. Statistical Verification
    # Calculate noise variance on the flat step section (t in [0.2s, 0.4s])
    steady_idx = (t >= 0.2) & (t <= 0.4)
    measured_input_std = np.std(noisy_signal[steady_idx] - clean_signal[steady_idx])
    measured_output_std = np.std(output_buf.data[steady_idx] - clean_signal[steady_idx])
    expected_output_std = sigma_noise / np.sqrt(M)
    
    print("=== MOVING AVERAGE FILTER METRICS ===")
    print(f"Window Size (M):               {M}")
    print(f"Measured Input Noise Std Dev:  {measured_input_std:.4f}")
    print(f"Measured Output Noise Std Dev: {measured_output_std:.4f}")
    print(f"Theoretical Output Noise Std:  {expected_output_std:.4f}")
    
    # 6. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    
    # Time-domain plot
    ax1.plot(t, noisy_signal, label='Noisy Input', color='lightgray', linewidth=1.0)
    ax1.plot(t, clean_signal, label='Clean Target', color='green', linestyle='--', linewidth=1.5)
    ax1.plot(t, output_buf.data.real, label=f'Filtered (M={M})', color='blue', linewidth=2.0)
    ax1.set_title(f"Moving Average Filter Time-Domain Response (M={M})", fontweight='bold')
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # Frequency response plot
    w, h = freqz(b=np.ones(M)/M, a=[1.0], worN=1024, fs=fs)
    ax2.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color='purple', linewidth=1.5)
    ax2.set_title("Frequency Response |H(f)| (dB)", fontweight='bold')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim([-40, 5])
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation_31()
```

---

## 4. Observations & Exploration Tasks

To further analyze the characteristics of moving average filters:

1. **Trade-off between Smoothing and Transient Lag:** Increase $M$ to $50$. Observe how the filtered output becomes extremely smooth, but the step transition experiences a noticeable group delay of $\frac{M-1}{2} = 24.5$ samples.
2. **Cascaded Moving Average Filters:** Connect two identical $M$-point moving average blocks in series. Notice how cascading multiple boxcar filters produces a triangular/Gaussian-like impulse response (by the Central Limit Theorem), drastically reducing high-frequency stopband sidelobes.