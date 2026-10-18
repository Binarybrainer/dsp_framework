# 📦 SignalBuffer Data Structure Specification

In our early prototypes, we passed raw `numpy.ndarray` objects between DSP blocks. While fast, raw arrays lack context. A filter block doesn't know the sampling rate, and a channel model doesn't know the current time offset of the data chunk it is processing.

To solve this, we introduce the `SignalBuffer`. This class uses **Composition** (wrapping a NumPy array) rather than inheritance, providing a clean metadata layer while maintaining the zero-copy memory benefits required for our Hybrid Performance Strategy.

## 1. Core Architecture & Motivation

The `SignalBuffer` must track:
* **`data`**: The raw `numpy.ndarray` (usually `np.complex64` or `np.complex128`).
* **`sample_rate` ($f_s$)**: The sampling rate in Hz. Crucial for filters, PLLs, and frequency offsets.
* **`timestamp` ($t_0$)**: The time in seconds of the *first* sample in the buffer. Essential for continuous streaming across block boundaries.
* **`active_length`**: The number of valid samples currently in the buffer.

## 2. Python Implementation

Here is the implementation of the `SignalBuffer` class. It includes utility methods for generating time vectors and copying metadata.

```python
import numpy as np
from typing import Optional, Union

class SignalBuffer:
    """
    A smart wrapper around NumPy arrays to carry DSP metadata (sample rate, time)
    through a processing pipeline.
    """
    
    def __init__(self, 
                 data: np.ndarray, 
                 sample_rate: float = 1.0, 
                 timestamp: float = 0.0):
        """
        Initialize the SignalBuffer.
        
        Args:
            data (np.ndarray): The 1D or 2D array of signal samples.
            sample_rate (float): The sampling frequency in Hz.
            timestamp (float): The absolute time of the first sample.
        """
        self.data: np.ndarray = np.asarray(data)
        self.sample_rate: float = sample_rate
        self.timestamp: float = timestamp

    @property
    def active_length(self) -> int:
        """Return the number of samples in the primary axis."""
        return self.data.shape[0]

    def time_vector(self) -> np.ndarray:
        """
        Generate a time vector matching the current data length.
        Useful for plotting or time-dependent channel models.
        """
        dt = 1.0 / self.sample_rate
        return self.timestamp + np.arange(self.active_length) * dt

    def clone_metadata(self, new_data: Optional[np.ndarray] = None) -> 'SignalBuffer':
        """
        Create a new SignalBuffer with the same metadata but different data.
        If new_data is not provided, it creates an empty buffer of the same shape.
        """
        if new_data is None:
            new_data = np.zeros_like(self.data)
            
        return SignalBuffer(
            data=new_data,
            sample_rate=self.sample_rate,
            timestamp=self.timestamp
        )

    def advance_time(self) -> None:
        """
        Advance the timestamp by the length of the current data buffer.
        Used when operating in a continuous streaming loop.
        """
        duration = self.active_length / self.sample_rate
        self.timestamp += duration

    def __repr__(self) -> str:
        return (f"SignalBuffer(length={self.active_length}, "
                f"fs={self.sample_rate}Hz, t0={self.timestamp}s, dtype={self.data.dtype})")
```

## 3. Modifying the Pipeline for SignalBuffer

With `SignalBuffer` introduced, we update our `DSPBlock` API so that `process()` accepts and returns a `SignalBuffer` instead of a raw `np.ndarray`. 

This makes blocks much smarter. For example, the `FrequencyOffsetChannel` no longer needs the user to pass the sample rate manually during initialization:

```python
class FrequencyOffsetChannel(DSPBlock):
    def __init__(self, freq_offset_hz: float = 1e3) -> None:
        super().__init__(freq_offset_hz=freq_offset_hz)
        self.reset()
        
    def reset(self) -> None:
        self.current_phase = 0.0

    def process(self, input_buf: SignalBuffer, **kwargs) -> SignalBuffer:
        fo = self._params['freq_offset_hz']
        
        # The block automatically knows the sample rate and time!
        t = input_buf.time_vector() 
        
        # Apply phase rotation
        phase_rotation = 2 * np.pi * fo * t + self.current_phase
        output_data = input_buf.data * np.exp(1j * phase_rotation)
        
        # Update state for next chunk
        self.current_phase = (self.current_phase + 2 * np.pi * fo * (input_buf.active_length / input_buf.sample_rate)) % (2 * np.pi)
        
        # Return a new SignalBuffer with cloned metadata
        return input_buf.clone_metadata(output_data)
```

## 4. Zero-Copy C++ Integration Readiness

Because we used composition (`self.data = np.asarray(data)`) instead of subclassing `np.ndarray`, passing this to C++ later via `pybind11` remains incredibly simple. 

When invoking a C++ backend, we simply pass `buffer.data` into the `pybind11` bound function, entirely bypassing the Python metadata overhead at the C++ level. The C++ function processes the raw array in-place, and the Python wrapper returns the updated `SignalBuffer`.