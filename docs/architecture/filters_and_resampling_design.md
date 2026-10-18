# 🎛️ Filters & Resampling Module Specification

Filtering and resampling are fundamental operations in any DSP framework. Since our framework focuses on continuous streaming (processing data chunk-by-chunk), the biggest challenge is **state preservation across block boundaries**. If a filter or resampler forgets its state between chunks, it will introduce severe transients (glitches) at the edges of every buffer.

This document specifies how to implement FIR/IIR filters, Root Raised Cosine (RRC) pulse shaping, and Arbitrary Fractional Resampling while perfectly maintaining boundary continuity.

## 1. General LTI Filter (FIR & IIR)

We leverage `scipy.signal.lfilter` for this block. To maintain state across `process()` calls, we use the `zi` (initial conditions) and `zf` (final conditions) arguments.

```python
import numpy as np
from scipy.signal import lfilter, lfiltic
from typing import Any, Union
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class LTI_FilterBlock(DSPBlock):
    """
    Applies a standard Linear Time-Invariant (LTI) filter.
    Can act as an FIR filter (a=[1.0]) or an IIR filter.
    """
    def __init__(self, b: np.ndarray, a: Union[np.ndarray, list] = [1.0]) -> None:
        # b: Numerator coefficients (Feedforward)
        # a: Denominator coefficients (Feedback)
        super().__init__(b=np.asarray(b), a=np.asarray(a))
        self.reset()
        
    def reset(self) -> None:
        """Initialize the delay line state (zi) to zeros."""
        b = self._params['b']
        a = self._params['a']
        # Compute the required state vector length
        state_length = max(len(b), len(a)) - 1
        
        if state_length > 0:
            # lfiltic computes the initial state required for lfilter
            self.filter_state = np.zeros(state_length, dtype=np.complex128)
        else:
            self.filter_state = None

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        b = self._params['b']
        a = self._params['a']
        
        if self.filter_state is not None:
            # Filter and update state
            out_data, self.filter_state = lfilter(
                b, a, input_buf.data, zi=self.filter_state
            )
        else:
            # Memoryless filtering (e.g., b=[1.0], a=[1.0])
            out_data = lfilter(b, a, input_buf.data)
            
        return input_buf.clone_metadata(out_data)
```

## 2. Root Raised Cosine (RRC) Pulse Shaping

The RRC filter is the industry standard for Transmitter Pulse Shaping and Receiver Matched Filtering. It prevents Intersymbol Interference (ISI). 
Instead of writing a new filtering loop, the `RRCFilterBlock` simply calculates the specific RRC taps during initialization and then behaves exactly like the `LTI_FilterBlock` (an FIR filter).

```python
class RRCFilterBlock(LTI_FilterBlock):
    """
    Root Raised Cosine Pulse Shaping Filter.
    Inherits from LTI_FilterBlock so it gets continuous streaming for free.
    """
    def __init__(self, alpha: float = 0.35, sps: int = 4, span_symbols: int = 10) -> None:
        """
        Args:
            alpha: Roll-off factor (0 to 1).
            sps: Samples per symbol.
            span_symbols: Length of the filter in symbols.
        """
        taps = self._generate_rrc_taps(alpha, sps, span_symbols)
        # Pass the calculated taps to the parent FIR filter class
        super().__init__(b=taps, a=[1.0])
        # Save RRC specific params in case we need to update them
        self._params.update({'alpha': alpha, 'sps': sps, 'span_symbols': span_symbols})

    def _generate_rrc_taps(self, alpha: float, sps: int, span: int) -> np.ndarray:
        """Generates the impulse response of the RRC filter."""
        # Time vector centered around zero
        t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
        
        # Handle the singularities (division by zero) mathematically
        taps = np.zeros(len(t), dtype=np.float64)
        for i, t_val in enumerate(t):
            if t_val == 0.0:
                taps[i] = 1.0 - alpha + (4 * alpha / np.pi)
            elif alpha != 0 and abs(t_val) == 1.0 / (4 * alpha):
                taps[i] = (alpha / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                num = np.sin(np.pi * t_val * (1 - alpha)) + \
                      4 * alpha * t_val * np.cos(np.pi * t_val * (1 + alpha))
                den = np.pi * t_val * (1 - (4 * alpha * t_val)**2)
                taps[i] = num / den
                
        # Normalize filter gain
        return taps / np.sqrt(np.sum(taps**2))
```

## 3. Fractional Resampler (Interpolation)

When a receiver clock is slightly mismatched from the transmitter, or when interfacing two hardware systems running at different rates, you must resample the signal by a fractional amount (e.g., from 100 MHz to 99.999 MHz).

We use `scipy.interpolate.interp1d`. To maintain continuity across chunks, we must buffer the very last sample of the previous chunk so the interpolator can draw a smooth line connecting the chunks.

```python
from scipy.interpolate import interp1d

class FractionalResamplerBlock(DSPBlock):
    """
    Changes the sampling rate of the signal by an arbitrary fractional factor
    using linear or cubic interpolation.
    """
    def __init__(self, target_fs: float, kind: str = 'linear') -> None:
        super().__init__(target_fs=target_fs, kind=kind)
        self.reset()
        
    def reset(self) -> None:
        # Buffer to hold the trailing edge of the previous chunk 
        # to ensure seamless boundary interpolation.
        self.overlap_time = None
        self.overlap_data = None

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        target_fs = self._params['target_fs']
        kind = self._params['kind']
        
        # Extract time and data from current chunk
        t_in = input_buf.time_vector()
        d_in = input_buf.data
        
        # Prepend overlap from previous chunk if it exists
        if self.overlap_time is not None:
            t_in = np.concatenate(([self.overlap_time], t_in))
            d_in = np.concatenate(([self.overlap_data], d_in))

        # Create continuous interpolator function
        interpolator = interp1d(t_in, d_in, kind=kind, assume_sorted=True)
        
        # Determine valid time range for resampling
        t_start = t_in[0]
        t_end = t_in[-1]
        
        # Generate new time points at the target sampling rate
        # We stop strictly before t_end to save the last sample for the next chunk
        dt_out = 1.0 / target_fs
        num_new_samples = int(np.floor((t_end - t_start) * target_fs))
        t_out = t_start + np.arange(num_new_samples) * dt_out
        
        # Resample!
        out_data = interpolator(t_out)
        
        # Save the absolute last sample from the INPUT stream for the next iteration
        self.overlap_time = t_in[-1]
        self.overlap_data = d_in[-1]
        
        # Return new SignalBuffer with updated sample rate and timestamp
        return SignalBuffer(
            data=out_data,
            sample_rate=target_fs,
            timestamp=t_out[0] if len(t_out) > 0 else input_buf.timestamp
        )
```

## 4. Pipeline Integration Summary

Because the `RRCFilterBlock` inherently inherits the `lfilter` delay line, and the `FractionalResamplerBlock` retains boundary overlaps, they can be flawlessly chained. 

```python
# 1. Create Blocks
tx_rrc = RRCFilterBlock(alpha=0.35, sps=4, span_symbols=15)
rx_resampler = FractionalResamplerBlock(target_fs=1.5e6) # Re-clock to 1.5 MHz

# 2. Chain into pipeline
system = DSPPipeline([tx_rrc, rx_resampler])

# 3. Stream data seamlessly
# Even if processed in 1024-sample chunks, no glitches will appear 
# in the time-domain or spectrum plots!
```