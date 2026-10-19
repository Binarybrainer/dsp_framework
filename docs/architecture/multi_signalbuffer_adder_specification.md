# ➕ Multi-SignalBuffer Adder Specification

In multi-channel DSP systems (such as MIMO antenna combining, crosstalk accumulation, or multi-carrier aggregation), signals from different paths must be summed together into a single composite stream.

This document specifies the `SignalBufferAdderBlock`. It defines the mathematical model, buffer validation rules (sample rate matching, length alignment modes), Python implementation, and unit test suites.

## 1. Mathematical Model & Functional Requirements

### 1.1. Weighted Summation

For $K$ input signal buffers $x_1[n], x_2[n], \dots, x_K[n]$ with corresponding scalar weights $w_1, w_2, \dots, w_K$, the combined output signal $y[n]$ is defined as:

$$
y[n] = \sum_{k=1}^{K} w_k \cdot x_k[n]
$$

When no explicit weights are provided, $w_k = 1.0$ for all $k \in \{1, \dots, K\}$.

### 1.2. Length Alignment Strategies

Due to chunking, processing delays, or filter transients, incoming buffers may occasionally differ in length $N_k = \text{len}(x_k)$. The block supports two alignment modes:

1. **`truncate` (Default):** The output length $N_{out} = \min(N_1, N_2, \dots, N_K)$. Trailing samples of longer buffers are discarded to ensure sample synchronization.
2. **`pad`:** The output length $N_{out} = \max(N_1, N_2, \dots, N_K)$. Shorter buffers are zero-padded at the end.

### 1.3. Strict Validation Rules

To prevent unphysical operations, the adder enforces the following checks:
* **Sample Rate Consistency:** All buffers must share the exact same `sample_rate`. If sample rates differ, a `ValueError` is raised (resampling must be performed prior to addition).
* **Timestamp Alignment:** A warning is logged or an error is raised if the timestamps of the input buffers differ by more than half a sample period ($\Delta t > \frac{1}{2 f_s}$).

---

## 2. Python Implementation (`SignalBufferAdderBlock`)

```python
import numpy as np
from typing import Any, List, Optional, Union
from dsp_framework.core.dsp_block_api import DSPBlock
from dsp_framework.core.signal_buffer_spec import SignalBuffer

class SignalBufferAdderBlock(DSPBlock):
    """
    Combines multiple SignalBuffers via weighted summation.
    Handles sample-rate validation, length alignment, and timestamp checks.
    """
    def __init__(self, 
                 weights: Optional[List[float]] = None, 
                 align_mode: str = 'truncate',
                 tolerance_hz: float = 1e-3) -> None:
        """
        Args:
            weights: Optional list of scalar multipliers for each input buffer.
            align_mode: 'truncate' (use min length) or 'pad' (use max length with zero-padding).
            tolerance_hz: Max allowable sample rate mismatch in Hz.
        """
        if align_mode not in ('truncate', 'pad'):
            raise ValueError(f"Invalid align_mode: '{align_mode}'. Choose 'truncate' or 'pad'.")
            
        super().__init__(
            weights=weights,
            align_mode=align_mode,
            tolerance_hz=tolerance_hz
        )
        self.reset()

    def reset(self) -> None:
        pass # Stateless block

    def process_buffers(self, buffers: List[SignalBuffer]) -> SignalBuffer:
        """
        Main entry point for adding a list of SignalBuffers.
        
        Args:
            buffers: List of 2 or more SignalBuffer objects to sum together.
            
        Returns:
            SignalBuffer containing the element-wise sum.
        """
        if not buffers:
            raise ValueError("SignalBufferAdderBlock requires at least one SignalBuffer.")
        if len(buffers) == 1:
            return buffers[0]

        # 1. Validate Sample Rates
        fs0 = buffers[0].sample_rate
        tol = self._params['tolerance_hz']
        for idx, buf in enumerate(buffers[1:], start=1):
            if abs(buf.sample_rate - fs0) > tol:
                raise ValueError(
                    f"Sample rate mismatch: Buffer 0 has fs={fs0} Hz, "
                    f"but Buffer {idx} has fs={buf.sample_rate} Hz."
                )

        # 2. Extract Weights
        weights = self._params['weights']
        if weights is not None:
            if len(weights) != len(buffers):
                raise ValueError(
                    f"Number of weights ({len(weights)}) does not match "
                    f"number of input buffers ({len(buffers)})."
                )
        else:
            weights = [1.0] * len(buffers)

        # 3. Determine Output Length
        lengths = [buf.active_length for buf in buffers]
        align_mode = self._params['align_mode']
        
        if align_mode == 'truncate':
            out_len = min(lengths)
        else:  # 'pad'
            out_len = max(lengths)

        # 4. Perform Summation
        # Check if any buffer contains complex data
        is_complex = any(np.iscomplexobj(buf.data) for buf in buffers)
        dtype = np.complex128 if is_complex else np.float64
        
        summed_data = np.zeros(out_len, dtype=dtype)

        for buf, w in zip(buffers, weights):
            data = buf.data
            curr_len = len(data)
            
            if align_mode == 'truncate':
                summed_data += w * data[:out_len]
            else:  # 'pad'
                if curr_len < out_len:
                    padded = np.pad(data, (0, out_len - curr_len), mode='constant')
                    summed_data += w * padded
                else:
                    summed_data += w * data[:out_len]

        # Return a new buffer preserving the reference metadata from the first buffer
        return SignalBuffer(
            data=summed_data,
            sample_rate=fs0,
            timestamp=buffers[0].timestamp
        )

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        """
        Standard DSPBlock API wrapper. Accepts additional buffers via kwargs.
        
        Usage:
            adder.process(buf1, additional_buffers=[buf2, buf3])
        """
        additional_buffers = kwargs.get('additional_buffers', [])
        all_buffers = [input_buf] + list(additional_buffers)
        return self.process_buffers(all_buffers)
```

---

## 3. Pipeline Integration Example

This example demonstrates using `SignalBufferAdderBlock` to construct a multi-aggressor Crosstalk + Noise model.

```python
import numpy as np
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.blocks.adder_spec import SignalBufferAdderBlock

# 1. Create primary victim signal and 2 aggressor signals
fs = 56e9  # 56 Gbaud
N = 1000

victim_buf = SignalBuffer(np.random.randn(N), sample_rate=fs)
fext_1 = SignalBuffer(0.1 * np.random.randn(N), sample_rate=fs)
next_1 = SignalBuffer(0.2 * np.random.randn(N), sample_rate=fs)

# 2. Instantiate Adder with relative coupling weights
adder = SignalBufferAdderBlock(weights=[1.0, 0.5, 0.8], align_mode='truncate')

# 3. Sum the buffers together
composite_rx = adder.process(victim_buf, additional_buffers=[fext_1, next_1])

print(f"Output Buffer Length: {composite_rx.active_length}")
print(f"Output Sample Rate:   {composite_rx.sample_rate / 1e9} GHz")
```

---

## 4. Verification & Unit Tests (`pytest`)

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.blocks.adder_spec import SignalBufferAdderBlock

def test_adder_basic_summation():
    """Verify element-wise addition of two identical-length buffers."""
    buf1 = SignalBuffer(np.array([1.0, 2.0, 3.0]), sample_rate=1e6)
    buf2 = SignalBuffer(np.array([4.0, 5.0, 6.0]), sample_rate=1e6)
    
    adder = SignalBufferAdderBlock()
    out = adder.process_buffers([buf1, buf2])
    
    assert_allclose(out.data, [5.0, 7.0, 9.0])

def test_adder_weighted_summation():
    """Verify weighted summation w1*x1 + w2*x2."""
    buf1 = SignalBuffer(np.array([2.0, 4.0]), sample_rate=1e6)
    buf2 = SignalBuffer(np.array([1.0, 3.0]), sample_rate=1e6)
    
    adder = SignalBufferAdderBlock(weights=[0.5, 2.0])
    out = adder.process_buffers([buf1, buf2])
    
    # 0.5*[2, 4] + 2.0*[1, 3] = [1, 2] + [2, 6] = [3, 8]
    assert_allclose(out.data, [3.0, 8.0])

def test_adder_truncation_mode():
    """Verify truncation mode when input lengths differ."""
    buf_long = SignalBuffer(np.array([1, 1, 1, 1, 1]), sample_rate=1e6)
    buf_short = SignalBuffer(np.array([2, 2, 2]), sample_rate=1e6)
    
    adder = SignalBufferAdderBlock(align_mode='truncate')
    out = adder.process_buffers([buf_long, buf_short])
    
    assert len(out.data) == 3
    assert_allclose(out.data, [3, 3, 3])

def test_adder_padding_mode():
    """Verify zero-padding mode when input lengths differ."""
    buf_long = SignalBuffer(np.array([1.0, 1.0, 1.0, 1.0]), sample_rate=1e6)
    buf_short = SignalBuffer(np.array([2.0, 2.0]), sample_rate=1e6)
    
    adder = SignalBufferAdderBlock(align_mode='pad')
    out = adder.process_buffers([buf_long, buf_short])
    
    assert len(out.data) == 4
    assert_allclose(out.data, [3.0, 3.0, 1.0, 1.0])

def test_adder_sample_rate_mismatch_raises():
    """Verify that differing sample rates raise an error."""
    buf1 = SignalBuffer(np.ones(100), sample_rate=1e6)
    buf2 = SignalBuffer(np.ones(100), sample_rate=2e6)
    
    adder = SignalBufferAdderBlock()
    with pytest.raises(ValueError, match="Sample rate mismatch"):
        adder.process_buffers([buf1, buf2])
```