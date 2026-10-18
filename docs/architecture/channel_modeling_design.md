# 📡 Channel Modeling as DSP Blocks

In a modular DSP framework, the physical transmission medium (the "Channel") is treated exactly like any other digital signal processing block. It inherits from the `DSPBlock` base class and can be inserted directly into a `DSPPipeline`. 

Instead of filtering or recovering a signal, a Channel Block mathematically applies impairments such as noise, phase rotation, and multipath echoes. 

This document outlines how to implement three essential channel models: **AWGN**, **Frequency Offset**, and **Multipath Fading**.

## 1. Additive White Gaussian Noise (AWGN) Channel

The most fundamental channel model. It adds complex Gaussian noise to the signal based on a target Signal-to-Noise Ratio (SNR). Because AWGN is memoryless, the `reset()` method doesn't need to clear any historical state.

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock

class AWGNChannel(DSPBlock):
    """
    Adds Additive White Gaussian Noise to a complex baseband signal.
    """
    def __init__(self, snr_db: float = 20.0) -> None:
        super().__init__(snr_db=snr_db)
        
    def reset(self) -> None:
        pass # AWGN is memoryless

    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        snr_linear = 10 ** (self._params['snr_db'] / 10.0)
        
        # Calculate average signal power
        signal_power = np.mean(np.abs(input_data)**2)
        
        # Calculate required noise variance
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        # Divide by 2 because power is split between I and Q components
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*input_data.shape) + 1j * np.random.randn(*input_data.shape)
        )
        
        return input_data + noise
```

## 2. Frequency Offset (Carrier Mismatch) Channel

This block simulates a mismatch between the transmitter and receiver local oscillators, causing the received constellation to rotate continuously over time. 

Unlike AWGN, this block **requires state management**. The phase must be preserved between consecutive calls to `process()` to ensure the rotation doesn't "jump" at the boundary of two data frames.

```python
class FrequencyOffsetChannel(DSPBlock):
    """
    Applies a continuous frequency offset (phase rotation) to the signal.
    """
    def __init__(self, freq_offset_hz: float = 1e3, sample_rate_hz: float = 1e6) -> None:
        super().__init__(freq_offset_hz=freq_offset_hz, sample_rate_hz=sample_rate_hz)
        self.reset()
        
    def reset(self) -> None:
        """Reset the phase accumulator for a new simulation run."""
        self.current_phase = 0.0

    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        fo = self._params['freq_offset_hz']
        fs = self._params['sample_rate_hz']
        num_samples = len(input_data)
        
        # Generate time vector for the current block
        t = np.arange(num_samples) / fs
        
        # Calculate instantaneous phase rotation
        phase_rotation = 2 * np.pi * fo * t + self.current_phase
        
        # Update the state for the next block to ensure continuous rotation
        self.current_phase = (self.current_phase + 2 * np.pi * fo * (num_samples / fs)) % (2 * np.pi)
        
        # Apply rotation
        return input_data * np.exp(1j * phase_rotation)
```

## 3. Multipath Fading (FIR) Channel

A multipath channel simulates signal reflections bouncing off buildings or terrain. Mathematically, a static multipath channel is a Convolution (an FIR filter). We use `scipy.signal.lfilter` to handle the convolution efficiently while preserving the filter state (the "delay line") across block boundaries using `zi` and `zf`.

```python
from scipy.signal import lfilter, lfiltic

class StaticMultipathChannel(DSPBlock):
    """
    Simulates a static multipath fading channel using a tapped delay line (FIR).
    """
    def __init__(self, taps: np.ndarray = np.array([1.0, 0.5j, -0.2])) -> None:
        super().__init__(taps=taps)
        self.reset()
        
    def reset(self) -> None:
        """Clear the FIR filter delay line."""
        taps = self._params['taps']
        # lfilter expects state array of length max(len(a), len(b)) - 1
        # Denominator array 'a' is just [1.0] for FIR
        self.filter_state = np.zeros(len(taps) - 1, dtype=np.complex128)

    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        taps = self._params['taps']
        
        # Apply FIR filter (convolution). 
        # Pass the previous state in (zi) and save the new state out (zf).
        output_data, self.filter_state = lfilter(
            b=taps, 
            a=[1.0], 
            x=input_data, 
            zi=self.filter_state
        )
        
        return output_data
```

## 4. Pipeline Integration Example

Because they are all `DSPBlock` instances, you can chain them seamlessly in your pipeline to create a harsh, realistic wireless channel:

```python
# Create channel impairments
multipath = StaticMultipathChannel(taps=np.array([1.0, 0.4+0.1j, -0.2]))
freq_offset = FrequencyOffsetChannel(freq_offset_hz=500.0, sample_rate_hz=1e6)
awgn = AWGNChannel(snr_db=15.0)

# Build the overall channel pipeline
wireless_channel = DSPPipeline([multipath, freq_offset, awgn])

# Process the transmitted signal through the entire channel
rx_signal = wireless_channel.process(tx_signal)
```