# 🎯 Simulation 51: Zero-Forcing (ZF) Linear Equalization for a Known Channel

In high-speed digital communications, transmission channels with finite bandwidth or multipath propagation induce **Intersymbol Interference (ISI)**. A **Zero-Forcing (ZF) Equalizer** is a linear filter designed to invert the channel's frequency/time response, forcing the combined channel-equalizer impulse response to meet the Nyquist ISI criterion (i.e., zero ISI at the sampling instances).

This document details the matrix-based mathematical derivation of a time-domain FIR Zero-Forcing equalizer, its Python implementation as a `DSPBlock`, and a simulation testbench evaluating performance over a known ISI channel.

---

## 1. Mathematical Background

### 1.1. Channel Model and ISI

Consider a discrete-time channel with impulse response $\mathbf{h} = [h_0, h_1, \dots, h_{L-1}]^T$ of length $L$. The received signal $y[n]$ is given by:

$$
y[n] = (x * h)[n] + v[n] = \sum_{k=0}^{L-1} h_k x[n-k] + v[n]
$$

where $x[n]$ is the transmitted symbol sequence and $v[n] \sim \mathcal{CN}(0, \sigma_v^2)$ is Additive White Gaussian Noise (AWGN).

### 1.2. Equalizer Formulation

We seek an FIR equalizer $\mathbf{w} = [w_0, w_1, \dots, w_{N_w-1}]^T$ of length $N_w$. The equalized output $\hat{x}[n]$ is:

$$
\hat{x}[n] = (y * w)[n] = \sum_{m=0}^{N_w-1} w_m y[n-m]
$$

The overall impulse response of the cascaded channel and equalizer is $g[n] = (h * w)[n]$, which has length $L_{total} = L + N_w - 1$.

To eliminate ISI completely at a decision delay $D$ ($0 \le D < L + N_w - 1$), we force $g[n]$ to match a delayed Kronecker delta function $\delta[n - D]$:

$$
g[n] = (h * w)[n] = \delta[n - D] = \begin{cases} 1, & n = D \\ 0, & n \neq D \end{cases}
$$

### 1.3. Matrix Representation and Pseudo-Inverse Solution

The convolution $g[n] = (h * w)[n]$ can be expressed as a matrix-vector product:

$$
\mathbf{H} \mathbf{w} = \mathbf{d}
$$

where $\mathbf{H}$ is the $(N_w + L - 1) \times N_w$ **Channel Convolution Matrix**:

$$
\mathbf{H} = \begin{bmatrix} 
h_0 & 0 & \dots & 0 \\
h_1 & h_0 & \dots & 0 \\
\vdots & h_1 & \ddots & \vdots \\
h_{L-1} & \vdots & \ddots & h_0 \\
0 & h_{L-1} & \dots & h_1 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & h_{L-1}
\end{bmatrix}
$$

and $\mathbf{d}$ is the target vector of length $N_w + L - 1$ containing $1$ at index $D$ and $0$ elsewhere:

$$
\mathbf{d} = [0, \dots, 0, \underbrace{1}_{\text{index } D}, 0, \dots, 0]^T
$$

Since the system $\mathbf{H} \mathbf{w} = \mathbf{d}$ is overdetermined ($N_w + L - 1 > N_w$), an exact solution exists only if $L=1$. Otherwise, we compute the Minimum Mean Squared Error (MMSE) approximation to the Zero-Forcing condition via the **Moore-Penrose Pseudo-Inverse**:

$$
\mathbf{w}_{ZF} = (\mathbf{H}^H \mathbf{H})^{-1} \mathbf{H}^H \mathbf{d} = \mathbf{H}^\dagger \mathbf{d}
$$

### 1.4. The Noise Enhancement Penalty

While the Zero-Forcing equalizer completely eliminates ISI (or minimizes it for finite length $N_w$), it inverts the channel spectrum $H(f)$:

$$
W_{ZF}(f) \approx \frac{1}{H(f)}
$$

If the channel has deep spectral nulls ($H(f) \to 0$), the equalizer gain $|W_{ZF}(f)|$ becomes extremely large. The noise power at the output of the equalizer is scaled by the squared norm of the equalizer weights:

$$
\sigma_{noise, out}^2 = \sigma_v^2 \cdot \|\mathbf{w}_{ZF}\|^2 = \sigma_v^2 \sum_{m=0}^{N_w-1} |w_m|^2
$$

This phenomenon—**Noise Enhancement**—is the primary drawback of Zero-Forcing equalization and motivates the use of MMSE or DFE equalizers in low-SNR or deep-null channels.

---

## 2. Python Implementation (`ZeroForcingEqualizerBlock`)

```python
import numpy as np
from scipy.signal import lfilter, lfilter_zi
from typing import Any, Optional
from dsp_framework.core.dsp_block_api import DSPBlock
from dsp_framework.core.signal_buffer_spec import SignalBuffer

class ZeroForcingEqualizerBlock(DSPBlock):
    """
    Stateful Linear Zero-Forcing (ZF) FIR Equalizer.
    Calculates weights via channel convolution matrix pseudo-inverse.
    """
    def __init__(self, 
                 channel_taps: np.ndarray, 
                 num_taps: int = 21, 
                 delay: Optional[int] = None) -> None:
        """
        Args:
            channel_taps: Known channel impulse response h[n].
            num_taps: Number of FIR equalizer taps N_w (should be odd, e.g., 21).
            delay: Targeted decision delay D. If None, defaults to center delay (N_w + L - 2) // 2.
        """
        h = np.asarray(channel_taps, dtype=np.complex128)
        L = len(h)
        N_w = num_taps
        
        if delay is None:
            delay = (N_w + L - 2) // 2
            
        super().__init__(channel_taps=h, num_taps=N_w, delay=delay)
        self._compute_weights()
        self.reset()

    def _compute_weights(self) -> None:
        """Computes ZF weights using the channel convolution matrix pseudo-inverse."""
        h = self._params['channel_taps']
        N_w = self._params['num_taps']
        D = self._params['delay']
        L = len(h)
        
        total_len = N_w + L - 1
        
        # 1. Construct Channel Convolution Matrix H of size (total_len, N_w)
        H = np.zeros((total_len, N_w), dtype=np.complex128)
        for col in range(N_w):
            H[col : col + L, col] = h
            
        # 2. Construct Desired Target Vector d
        d = np.zeros(total_len, dtype=np.complex128)
        d[D] = 1.0
        
        # 3. Solve w_zf = pinv(H) @ d
        w_zf = np.linalg.pinv(H) @ d
        self.weights = w_zf

    def reset(self) -> None:
        """Reset internal filter memory for continuous streaming."""
        self.zi = lfilter_zi(b=self.weights, a=[1.0]) * 0.0

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        w = self.weights
        
        # Apply FIR equalization with state tracking across chunks
        if np.iscomplexobj(data):
            y_real, zi_real = lfilter(w.real, [1.0], data.real, zi=self.zi.real)
            y_imag, zi_imag = lfilter(w.real, [1.0], data.imag, zi=self.zi.imag)
            y = y_real + 1j * y_imag
            self.zi = zi_real + 1j * zi_imag
        else:
            y, self.zi = lfilter(w.real, [1.0], data, zi=self.zi)

        return input_buf.clone_metadata(y)
```

---

## 3. Simulation Testbench & Verification Script

The following simulation evaluates 16-QAM signal transmission through a multipath ISI channel, compares the constellation before and after Zero-Forcing equalization, and quantifies noise enhancement.

```python
import numpy as np
import matplotlib.pyplot as plt
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.blocks.channel_modeling_spec import AWGNChannel
from dsp_framework.blocks.modulation_demodulation_spec import SymbolMapperBlock
from dsp_framework.blocks.zf_equalizer import ZeroForcingEqualizerBlock

def run_simulation_51():
    # 1. Setup Parameters
    N_symbols = 5000
    fs = 1e6
    snr_db = 25.0
    
    # 2. Generate 16-QAM Signal
    mapper = SymbolMapperBlock(modulation_type='QAM', M=16)
    tx_symbols = np.random.randint(0, 16, N_symbols)
    tx_buf = mapper.process(SignalBuffer(tx_symbols, sample_rate=fs))
    
    # 3. Define Known Frequency-Selective Channel
    # Channel with moderate ISI: h = [0.2, 0.9, -0.3, 0.1]
    channel_h = np.array([0.2, 0.9, -0.3, 0.1], dtype=np.complex128)
    channel_h /= np.linalg.norm(channel_h) # Normalize channel energy to 1
    
    # Pass through Channel
    channel_output = np.convolve(tx_buf.data, channel_h, mode='full')[:N_symbols]
    chan_buf = SignalBuffer(channel_output, sample_rate=fs)
    
    # Add Noise
    awgn = AWGNChannel(snr_db=snr_db)
    noisy_buf = awgn.process(chan_buf)
    
    # 4. Zero-Forcing Equalization
    num_taps = 31
    zf_eq = ZeroForcingEqualizerBlock(channel_taps=channel_h, num_taps=num_taps)
    equalized_buf = zf_eq.process(noisy_buf)
    
    # Account for equalizer decision delay when plotting/evaluating
    delay = zf_eq._params['delay']
    aligned_rx = equalized_buf.data[delay : delay + N_symbols - delay]
    aligned_tx = tx_buf.data[: len(aligned_rx)]
    
    # 5. Performance Metrics
    noise_enhancement_factor = np.sum(np.abs(zf_eq.weights)**2)
    evm_before = np.sqrt(np.mean(np.abs(noisy_buf.data - tx_buf.data)**2))
    evm_after = np.sqrt(np.mean(np.abs(aligned_rx - aligned_tx)**2))
    
    print("=== ZERO-FORCING EQUALIZATION METRICS ===")
    print(f"Channel Impulse Response:    {channel_h}")
    print(f"Equalizer Taps (N_w):        {num_taps}")
    print(f"Equalizer Delay (D):         {delay}")
    print(f"Equalizer Noise Gain ||w||²: {noise_enhancement_factor:.3f} ({10*np.log10(noise_enhancement_factor):.2f} dB)")
    print(f"EVM Before Equalization:     {evm_before:.4f}")
    print(f"EVM After Equalization:      {evm_after:.4f}")

    # 6. Constellation Diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(noisy_buf.data.real, noisy_buf.data.imag, color='red', alpha=0.3, s=10)
    ax1.set_title("Before Equalization (Channel ISI + Noise)", fontweight='bold')
    ax1.set_xlabel("In-Phase")
    ax1.set_ylabel("Quadrature")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_aspect('equal')
    
    ax2.scatter(aligned_rx.real, aligned_rx.imag, color='blue', alpha=0.3, s=10)
    ax2.set_title(f"After ZF Equalization (N_w={num_taps})", fontweight='bold')
    ax2.set_xlabel("In-Phase")
    ax2.set_ylabel("Quadrature")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation_51()
```

---

## 4. Key Observations & Exploration Tasks

1. **ISI Removal vs. Noise Level:** At high SNR ($\ge 30\text{ dB}$), the Zero-Forcing equalizer completely opens the 16-QAM eye diagram and clusters points tightly around constellation nodes.
2. **Noise Enhancement at Low SNR:** Lower the SNR to $10\text{ dB}$ and observe the output constellation. Due to $\|\mathbf{w}_{ZF}\|^2 > 1$, noise is visibly amplified compared to an AWGN channel without ISI.
3. **Spectral Nulls:** Change `channel_h` to $[0.5, 0.5]$ (which produces a deep null at Nyquist frequency $f_s/2$). Observe how $\|\mathbf{w}_{ZF}\|^2$ skyrockets, drastically degrading the output SNR. This illustrates why **MMSE Equalization** (Simulation 52) is necessary in practical channels.