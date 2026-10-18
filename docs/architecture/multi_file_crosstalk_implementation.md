# 📁 Multi-File Crosstalk Channel Implementation

In many hardware measurement setups, Vector Network Analyzers (VNAs) or 3D EM solvers output standard 4-port Touchstone files (`.s4p`) for individual coupling paths rather than a single monolithic file (like `.s16p`). 

For a single victim lane, you might receive:
* **1 THRU file:** The main transmission path.
* **$N$ NEXT files:** Near-End Crosstalk from $N$ adjacent aggressor Tx pins.
* **$M$ FEXT files:** Far-End Crosstalk from $M$ distant aggressor Tx pins.

This document specifies the pipeline to parse multiple `.s4p` files, extract their time-domain Impulse Responses (IR), and aggregate them in an upgraded `MultiAggressorCrosstalkBlock`.

## 1. S-Parameter Extraction Strategy

When a crosstalk path is saved as its own `.s4p` file, the actual crosstalk transfer function is typically represented as the **differential forward transmission ($S_{dd21}$)** of *that specific file*. 
* For the THRU file, $S_{dd21}$ is the insertion loss.
* For a NEXT file, $S_{dd21}$ represents the isolation/coupling from the Aggressor Tx to the Victim Rx.

We reuse the Frequency-to-Time domain (IFFT) strategy defined in `s4p_channel_design.md` but apply it iteratively over lists of files.

### 1.1. Python Utility: Batch S-Parameter to FIR

```python
import skrf as rf
import numpy as np
from typing import List

def s4p_to_fir(filepath: str, sample_rate_hz: float, target_taps: int = 1024) -> np.ndarray:
    """Converts a single .s4p file's Sdd21 path to time-domain FIR taps."""
    network = rf.Network(filepath)
    network.se2gmm(p=2) # Convert to Mixed-Mode
    sdd21 = network.s[:, 1, 0] # Extract Differential S21
    freqs = network.f
    
    # Interpolate to DC-Nyquist grid
    target_freqs = np.linspace(0, sample_rate_hz / 2, target_taps // 2 + 1)
    sdd21_mag = np.interp(target_freqs, freqs, np.abs(sdd21))
    sdd21_phase = np.interp(target_freqs, freqs, np.unwrap(np.angle(sdd21)))
    sdd21_phase[0] = 0.0 # DC phase to 0
    
    H_target = sdd21_mag * np.exp(1j * sdd21_phase)
    taps = np.fft.irfft(H_target, n=target_taps) * np.hamming(target_taps)
    return taps

def load_crosstalk_profiles(thru_file: str, next_files: List[str], fext_files: List[str], fs: float) -> dict:
    """Batch processes multiple .s4p files into FIR arrays."""
    return {
        "thru": s4p_to_fir(thru_file, fs),
        "next": [s4p_to_fir(f, fs) for f in next_files],
        "fext": [s4p_to_fir(f, fs) for f in fext_files]
    }
```

## 2. Upgraded DSP Block: `MultiAggressorCrosstalkBlock`

We must upgrade our `CrosstalkChannelBlock` to handle a dynamic number of aggressors. It will now maintain a **list of delay lines** (one for each aggressor path) to ensure sample-by-sample continuity across chunks.

```python
from scipy.signal import lfilter
from typing import Any, Optional, List
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class MultiAggressorCrosstalkBlock(DSPBlock):
    """
    Simulates a channel with 1 THRU path and multiple independent NEXT/FEXT aggressors.
    """
    def __init__(self, 
                 thru_taps: np.ndarray, 
                 next_taps_list: List[np.ndarray] = [], 
                 fext_taps_list: List[np.ndarray] = []) -> None:
        
        super().__init__(
            thru_taps=np.asarray(thru_taps),
            next_taps_list=[np.asarray(t) for t in next_taps_list],
            fext_taps_list=[np.asarray(t) for t in fext_taps_list]
        )
        self.reset()

    def reset(self) -> None:
        """Initialize separate delay lines for every single tap profile."""
        self.state_thru = np.zeros(len(self._params['thru_taps']) - 1, dtype=np.complex128)
        
        # Lists of states for multiple aggressors
        self.states_next = [np.zeros(len(taps) - 1, dtype=np.complex128) 
                            for taps in self._params['next_taps_list']]
        
        self.states_fext = [np.zeros(len(taps) - 1, dtype=np.complex128) 
                            for taps in self._params['fext_taps_list']]

    def _generate_dummy_aggressor(self, length: int) -> np.ndarray:
        """Generates uncorrelated random noise to simulate an active adjacent lane."""
        return (np.random.randn(length) + 1j * np.random.randn(length)) / np.sqrt(2)

    def process(self, 
                input_buf: SignalBuffer, 
                next_signals: Optional[List[SignalBuffer]] = None, 
                fext_signals: Optional[List[SignalBuffer]] = None,
                **kwargs: Any) -> SignalBuffer:
        
        N = input_buf.active_length
        
        # 1. Process Main THRU Signal
        y_total, self.state_thru = lfilter(
            b=self._params['thru_taps'], a=[1.0], 
            x=input_buf.data, zi=self.state_thru
        )
        
        # 2. Accumulate NEXT from all aggressors
        for i, taps in enumerate(self._params['next_taps_list']):
            # Use provided signal or generate dummy uncorrelated data
            x_aggressor = next_signals[i].data if (next_signals and i < len(next_signals)) else self._generate_dummy_aggressor(N)
            
            y_next, self.states_next[i] = lfilter(b=taps, a=[1.0], x=x_aggressor, zi=self.states_next[i])
            y_total += y_next
            
        # 3. Accumulate FEXT from all aggressors
        for i, taps in enumerate(self._params['fext_taps_list']):
            x_aggressor = fext_signals[i].data if (fext_signals and i < len(fext_signals)) else self._generate_dummy_aggressor(N)
            
            y_fext, self.states_fext[i] = lfilter(b=taps, a=[1.0], x=x_aggressor, zi=self.states_fext[i])
            y_total += y_fext
            
        return input_buf.clone_metadata(y_total)
```

## 3. Usage Example in Pipeline

This architecture makes it incredibly easy to configure complex testbenches directly from a directory of `.s4p` exports.

```python
# 1. Load profiles from VNA exports
fs = 56e9 # 56 Gbaud
profiles = load_crosstalk_profiles(
    thru_file="data/lane0_thru.s4p",
    next_files=["data/lane0_next_aggressor1.s4p", "data/lane0_next_aggressor2.s4p"],
    fext_files=["data/lane0_fext_aggressor1.s4p"],
    fs=fs
)

# 2. Initialize the block
multi_crosstalk = MultiAggressorCrosstalkBlock(
    thru_taps=profiles["thru"],
    next_taps_list=profiles["next"],
    fext_taps_list=profiles["fext"]
)

# 3. Process
# The block will automatically generate dummy data for the 2 NEXT and 1 FEXT lanes
# while accurately preserving the filter states across chunk boundaries!
rx_signal = multi_crosstalk.process(tx_signal)
```