# 📡 Signal Generators Design (Sources)

In our DSP framework, Signal Generators act as the "Source" blocks at the very beginning of the pipeline. Unlike standard `DSPBlock` instances that process an input signal, Source blocks generate data from scratch.

To maintain compatibility with our streaming architecture and the `SignalBuffer` data structure, we will introduce a `SignalSource` base pattern. These blocks will maintain their own internal timestamp to ensure continuous, glitch-free waveform generation across multiple chunks.

## 1. Source Block Architecture

We adapt the `DSPBlock` interface slightly for sources. Instead of a `process()` method taking an input array, sources use a `generate()` method that takes the requested number of samples and the sampling rate.

```python
from typing import Any
import numpy as np
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class SignalSource(DSPBlock):
    """Base class for all signal generators."""
    def __init__(self, sample_rate_hz: float, **kwargs: Any) -> None:
        super().__init__(sample_rate_hz=sample_rate_hz, **kwargs)
        self.reset()
        
    def reset(self) -> None:
        """Reset the internal timestamp for continuous generation."""
        self.current_time = 0.0

    def generate(self, num_samples: int) -> SignalBuffer:
        """
        Generate a SignalBuffer of the requested length.
        Must be implemented by child classes.
        """
        pass
```

## 2. PRBS (Pseudo-Random Binary Sequence) Generator

PRBS is the industry standard for testing communication links (e.g., PRBS7, PRBS15, PRBS31). Instead of writing a slow LFSR (Linear Feedback Shift Register) loop in Python, we can leverage `scipy.signal.max_len_seq` to generate mathematically perfect PRBS sequences instantly.

```python
from scipy.signal import max_len_seq

class PRBSGenerator(SignalSource):
    """
    Generates a Pseudo-Random Binary Sequence (PRBS).
    Outputs a SignalBuffer containing binary data (0s and 1s).
    """
    def __init__(self, sample_rate_hz: float, prbs_order: int = 7) -> None:
        """
        prbs_order: e.g., 7 for PRBS7 (period = 2^7 - 1), 15 for PRBS15.
        """
        super().__init__(sample_rate_hz=sample_rate_hz, prbs_order=prbs_order)
        self._generate_full_sequence()
        
    def _generate_full_sequence(self) -> None:
        """Pre-generate the entire PRBS period to loop through."""
        order = self._params['prbs_order']
        # scipy max_len_seq returns the sequence and the state. We only need the sequence.
        self.full_seq, _ = max_len_seq(order)
        self.seq_length = len(self.full_seq)
        self.seq_index = 0

    def update_params(self, **kwargs: Any) -> None:
        super().update_params(**kwargs)
        if 'prbs_order' in kwargs:
            self._generate_full_sequence()

    def generate(self, num_samples: int) -> SignalBuffer:
        # Repeat the pre-generated sequence to satisfy the requested length
        repeats = (num_samples // self.seq_length) + 2
        tiled_seq = np.tile(self.full_seq, repeats)
        
        # Extract the exact chunk we need
        out_data = tiled_seq[self.seq_index : self.seq_index + num_samples]
        
        # Update index for the next chunk
        self.seq_index = (self.seq_index + num_samples) % self.seq_length
        
        # Create SignalBuffer
        buf = SignalBuffer(
            data=out_data.astype(np.float32), 
            sample_rate=self._params['sample_rate_hz'],
            timestamp=self.current_time
        )
        
        # Advance internal time
        self.current_time += num_samples / self._params['sample_rate_hz']
        return buf
```

## 3. Basic Waveforms (Sine, Cosine, Square)

These continuous-time waveforms rely heavily on the internal `current_time` state to ensure phase continuity. If we request 1000 samples now and 1000 samples later, the sine wave must not skip or jump.

```python
from scipy.signal import square

class WaveformGenerator(SignalSource):
    """
    Generates basic continuous waveforms: Sine, Cosine, or Square.
    """
    def __init__(self, 
                 sample_rate_hz: float, 
                 frequency_hz: float = 1e3, 
                 amplitude: float = 1.0,
                 phase_rad: float = 0.0,
                 wave_type: str = 'sine') -> None:
        
        super().__init__(
            sample_rate_hz=sample_rate_hz,
            frequency_hz=frequency_hz,
            amplitude=amplitude,
            phase_rad=phase_rad,
            wave_type=wave_type
        )

    def generate(self, num_samples: int) -> SignalBuffer:
        fs = self._params['sample_rate_hz']
        f0 = self._params['frequency_hz']
        amp = self._params['amplitude']
        phi = self._params['phase_rad']
        w_type = self._params['wave_type'].lower()

        # Generate continuous time vector
        dt = 1.0 / fs
        t = self.current_time + np.arange(num_samples) * dt
        
        # Calculate angular phase
        theta = 2 * np.pi * f0 * t + phi

        # Generate specific waveform
        if w_type == 'sine':
            data = amp * np.sin(theta)
        elif w_type == 'cosine':
            data = amp * np.cos(theta)
        elif w_type == 'square':
            # scipy.signal.square takes angular phase
            data = amp * square(theta)
        else:
            raise ValueError(f"Unknown wave_type: {w_type}")

        # Create SignalBuffer
        buf = SignalBuffer(
            data=data.astype(np.complex64),
            sample_rate=fs,
            timestamp=self.current_time
        )
        
        # Update internal state
        self.current_time += num_samples * dt
        
        return buf
```

## 4. Pipeline Integration Example

Because sources output a standard `SignalBuffer`, they feed perfectly into the rest of your channel and processing blocks.

```python
# 1. Initialize Source and Channel
fs = 1e6 # 1 MHz sampling rate
sine_source = WaveformGenerator(sample_rate_hz=fs, frequency_hz=10e3, wave_type='sine')
channel_noise = AWGNChannel(snr_db=10.0)

# 2. Setup a processing pipeline (without the source, just the processors)
rx_pipeline = DSPPipeline([channel_noise])

# 3. Simulate continuous streaming in chunks
chunk_size = 4096

for _ in range(5):
    # Source generates a clean chunk with precise timestamp
    clean_signal = sine_source.generate(chunk_size)
    
    # Pipeline adds impairments
    noisy_signal = rx_pipeline.process(clean_signal)
    
    print(noisy_signal)
    # Output: SignalBuffer(length=4096, fs=1000000.0Hz, t0=..., dtype=complex64)
```