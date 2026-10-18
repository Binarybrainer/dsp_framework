# 📡 Advanced Channel Modeling Specification

In a realistic DSP framework, the channel is rarely a static filter. Local oscillators introduce phase noise, devices move (causing Doppler shift and fading), and thermal noise degrades the SNR.

This document implements AWGN, Phase Noise, Frequency Offset, and Dynamic Fading (Rayleigh/Rician) blocks. All blocks natively accept and return the `SignalBuffer` object to maintain sample rate and timestamp integrity.

## 1. Additive White Gaussian Noise (AWGN)

The AWGN block adds thermal noise based on a target Signal-to-Noise Ratio ($E_s/N_0$ or SNR). Since it is memoryless, it doesn't require state preservation between chunks.

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class AWGNChannel(DSPBlock):
    """
    Applies Additive White Gaussian Noise to a complex signal.
    """
    def __init__(self, snr_db: float = 20.0) -> None:
        super().__init__(snr_db=snr_db)
        self.reset()
        
    def reset(self) -> None:
        pass # Memoryless

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        snr_linear = 10 ** (self._params['snr_db'] / 10.0)
        data = input_buf.data
        
        # Calculate signal power
        signal_power = np.mean(np.abs(data)**2)
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape)
        )
        
        out_data = data + noise
        return input_buf.clone_metadata(out_data)

```

## 2. Oscillator Impairments: Frequency Offset & Phase Noise

A real-world Local Oscillator (LO) has two main imperfections:
1. **Frequency Offset (CFO):** A deterministic, constant rotation.
2. **Phase Noise:** A random walk of the phase caused by the oscillator's physical linewidth. We model this as a Wiener process (integrated white noise).

```python
class OscillatorImpairmentChannel(DSPBlock):
    """
    Simulates Carrier Frequency Offset (CFO) and Phase Noise (Wiener process).
    Maintains phase continuity across processing chunks.
    """
    def __init__(self, freq_offset_hz: float = 1e3, linewidth_hz: float = 0.0) -> None:
        """
        freq_offset_hz: Constant frequency drift.
        linewidth_hz: Oscillator linewidth determining phase noise severity.
        """
        super().__init__(freq_offset_hz=freq_offset_hz, linewidth_hz=linewidth_hz)
        self.reset()
        
    def reset(self) -> None:
        self.current_phase = 0.0

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        fs = input_buf.sample_rate
        N = len(data)
        
        fo = self._params['freq_offset_hz']
        linewidth = self._params['linewidth_hz']
        
        # 1. Deterministic Frequency Offset
        # Using time_vector ensures absolute time accuracy
        t = input_buf.time_vector()
        phase_det = 2 * np.pi * fo * t
        
        # 2. Random Phase Noise (Wiener Process)
        if linewidth > 0:
            # Variance of phase step per sample = 2 * pi * linewidth / fs
            sigma_phase = np.sqrt(2 * np.pi * linewidth / fs)
            phase_steps = np.random.normal(0, sigma_phase, N)
            
            # Integrate steps to get random walk, add previous state
            phase_rand = np.cumsum(phase_steps) + self.current_phase
            
            # Save the last phase for the next chunk
            self.current_phase = phase_rand[-1]
        else:
            phase_rand = np.zeros(N)
            self.current_phase = 0.0 # Not needed if no phase noise
            
        # Total phase rotation
        total_phase = phase_det + phase_rand
        
        out_data = data * np.exp(1j * total_phase)
        return input_buf.clone_metadata(out_data)

```

## 3. Dynamic Fading Channel: Rayleigh & Rician

Unlike the static FIR multipath channel we discussed earlier, wireless mobile channels suffer from **Fading**. The channel taps change over time due to the receiver's movement (Doppler effect).

* **Rayleigh:** No Line-of-Sight (LoS) path. The envelope follows a Rayleigh distribution.
* **Rician:** A strong Line-of-Sight path exists alongside scattered paths. Governed by the K-factor.

We simulate a "Flat Fading" channel using a first-order Autoregressive (AR1) model to approximate the Doppler spread.

```python
from scipy.special import j0

class FlatFadingChannel(DSPBlock):
    """
    Simulates a time-varying Flat Fading channel (Rayleigh or Rician) 
    with a specific Doppler frequency.
    """
    def __init__(self, doppler_hz: float = 100.0, k_factor_linear: float = 0.0) -> None:
        """
        doppler_hz: Maximum Doppler shift (depends on velocity and carrier freq).
        k_factor_linear: Ratio of LoS power to scattered power (0 = Rayleigh).
        """
        super().__init__(doppler_hz=doppler_hz, k_factor_linear=k_factor_linear)
        self.reset()
        
    def reset(self) -> None:
        # Initial channel state (complex)
        self.h_state = 1.0 + 0j

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        fs = input_buf.sample_rate
        N = len(data)
        
        fd = self._params['doppler_hz']
        K = self._params['k_factor_linear']
        
        # Power normalization factors
        los_amplitude = np.sqrt(K / (K + 1))
        scatter_amplitude = np.sqrt(1 / (K + 1))
        
        if fd > 0:
            # AR1 Model parameters for Jakes/Doppler approximation
            # Autocorrelation coefficient for one sample delay
            rho = j0(2 * np.pi * fd / fs)
            noise_gain = np.sqrt(1 - rho**2)
            
            # Generate complex Gaussian noise for the scattered paths
            w = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
            
            # Time-varying channel coefficient array
            h_scatter = np.zeros(N, dtype=np.complex128)
            
            # AR1 Loop (Note: This is a fast Python loop because it only 
            # computes the channel coefficient, not an entire filter convolution)
            current_h = self.h_state
            for i in range(N):
                current_h = rho * current_h + noise_gain * w[i]
                h_scatter[i] = current_h
                
            self.h_state = current_h # Save state for next chunk
        else:
            # Static channel if no Doppler
            h_scatter = np.ones(N, dtype=np.complex128)

        # Combine Line-of-Sight (LoS is assumed constant phase here for simplicity) 
        # with the scattered fading
        h_total = los_amplitude + scatter_amplitude * h_scatter
        
        # Apply the flat fading coefficient to the signal
        out_data = data * h_total
        
        return input_buf.clone_metadata(out_data)

```

## 4. Pipeline Integration

You can now cascade these to create a highly realistic mobile wireless environment.

```python
# Create the environment blocks
fading = FlatFadingChannel(doppler_hz=120.0, k_factor_linear=2.0) # Rician fading
osc_noise = OscillatorImpairmentChannel(freq_offset_hz=200, linewidth_hz=1e4)
thermal_noise = AWGNChannel(snr_db=15.0)

# Build the channel pipeline
mobile_channel = DSPPipeline([fading, osc_noise, thermal_noise])

# The pipeline inherently handles all the states (phase, Doppler AR1) seamlessly!
rx_buf = mobile_channel.process(tx_buf)

```