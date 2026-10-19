# 🌊 Simulation 1: Continuous Waveform Generation

This document covers the first exercise in the DSP Simulation Checklist. Generating fundamental continuous-time waveforms (Sine, Cosine, and Square) is the foundational step for testing any DSP framework. These signals serve as test vectors for filters, modulators, and spectral analysis tools.

## 1. Mathematical Background

A continuous waveform can be defined by its amplitude ($A$), frequency ($f_0$ in Hz), and initial phase ($\phi$ in radians). In a digital system, time $t$ is discretized based on the sample rate ($f_s$), such that $t = n \cdot T_s = n / f_s$.

*   **Sine Wave:** $x[n] = A \sin(2\pi f_0 n / f_s + \phi)$
*   **Cosine Wave:** $x[n] = A \cos(2\pi f_0 n / f_s + \phi)$
*   **Square Wave:** $x[n] = A \cdot \text{sgn}(\sin(2\pi f_0 n / f_s + \phi))$

## 2. Python Implementation

We implement a `WaveformGenerator` class. Unlike standard `DSPBlock`s that take an input buffer, this acts as a *Source*, generating a new `SignalBuffer` from scratch and keeping track of its own continuous time state.

```python
import numpy as np
from scipy.signal import square
from dsp_framework.core.signal_buffer_spec import SignalBuffer

class WaveformGenerator:
    """
    Generates basic continuous waveforms: Sine, Cosine, or Square.
    Maintains internal time state for seamless chunked generation.
    """
    def __init__(self, 
                 sample_rate_hz: float, 
                 frequency_hz: float = 1e3, 
                 amplitude: float = 1.0,
                 phase_rad: float = 0.0,
                 wave_type: str = 'sine') -> None:
        
        self.sample_rate = sample_rate_hz
        self.frequency = frequency_hz
        self.amplitude = amplitude
        self.phase = phase_rad
        self.wave_type = wave_type.lower()
        self.current_time = 0.0

    def reset(self) -> None:
        """Resets the internal time counter to zero."""
        self.current_time = 0.0

    def generate(self, num_samples: int) -> SignalBuffer:
        """Generates a block of waveform samples."""
        dt = 1.0 / self.sample_rate
        t = self.current_time + np.arange(num_samples) * dt
        
        # Calculate angular phase
        theta = 2 * np.pi * self.frequency * t + self.phase

        # Generate specific waveform
        if self.wave_type == 'sine':
            data = self.amplitude * np.sin(theta)
        elif self.wave_type == 'cosine':
            data = self.amplitude * np.cos(theta)
        elif self.wave_type == 'square':
            data = self.amplitude * square(theta)
        else:
            raise ValueError(f"Unknown wave_type: {self.wave_type}")

        # Update internal state for the next chunk
        self.current_time += num_samples * dt
        
        return SignalBuffer(
            data=data.astype(np.complex128), # Cast to complex for framework consistency
            sample_rate=self.sample_rate,
            timestamp=t[0]
        )
```

## 3. Simulation Script & Visualization

This script tests the generator by creating one of each waveform and plotting them using our `MatplotlibDSPPlotter`.

```python
import matplotlib.pyplot as plt
from dsp_framework.blocks.waveform_generator import WaveformGenerator
from dsp_framework.utils.matplotlib_visualization_spec import MatplotlibDSPPlotter

def run_simulation_01():
    fs = 100e3  # 100 kHz sample rate
    f0 = 5e3    # 5 kHz signal frequency
    N = 100     # Generate 100 samples (2 periods)

    # 1. Initialize Generators
    sine_gen = WaveformGenerator(fs, f0, wave_type='sine')
    cos_gen = WaveformGenerator(fs, f0, wave_type='cosine')
    square_gen = WaveformGenerator(fs, f0, wave_type='square')

    # 2. Generate Signals
    sine_buf = sine_gen.generate(N)
    cos_buf = cos_gen.generate(N)
    square_buf = square_gen.generate(N)

    # 3. Plotting
    plotter = MatplotlibDSPPlotter()
    
    fig1 = plotter.plot_time_domain(sine_buf, title="5 kHz Sine Wave")
    fig2 = plotter.plot_time_domain(cos_buf, title="5 kHz Cosine Wave")
    fig3 = plotter.plot_time_domain(square_buf, title="5 kHz Square Wave")
    
    plt.show()

if __name__ == "__main__":
    run_simulation_01()
```

## 4. Exploration Tasks

To fully understand this block, try the following modifications in your script:
1.  **Aliasing Test:** Change `f0` to `60e3` (which is greater than the Nyquist frequency of $f_s / 2 = 50\text{kHz}$). Observe how the plotted frequency appears as $40\text{kHz}$ due to aliasing.
2.  **Streaming Continuity:** Call `generate(50)` twice in a row on the same generator and concatenate the outputs. Verify that there is no phase jump at the boundary between the two chunks.