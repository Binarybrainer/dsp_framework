# 🔀 Crosstalk Channel Modeling (THRU, NEXT, FEXT)

In high-speed digital links (such as SerDes, PCIe, or multi-lane Ethernet), the physical channel consists of multiple closely routed traces or cables. A receiver does not just see the attenuated main signal; it also sees interference coupling from adjacent lanes.

This document specifies the architecture for a `CrosstalkChannelBlock` capable of simulating **THRU**, **NEXT**, and **FEXT** simultaneously.

## 1. Definitions and Mathematical Model

*   **THRU (Through / Insertion Loss):** The main transmission path from the Victim Transmitter to the Victim Receiver.
*   **FEXT (Far-End Crosstalk):** Interference coupled from an Aggressor Transmitter located at the *same end* as the Victim Transmitter. The interference travels the full length of the channel alongside the victim signal.
*   **NEXT (Near-End Crosstalk):** Interference coupled from an Aggressor Transmitter located at the *same end* as the Victim Receiver. This interference travels backward relative to the victim signal and usually suffers less attenuation, making it highly destructive.

### The Convolution Model

If $h_{thru}$, $h_{next}$, and $h_{fext}$ are the time-domain impulse responses (FIR taps) of the respective paths, the total received signal $y[n]$ is:

$$y[n] = (x_{vic} * h_{thru})[n] + \sum_{i} (x_{fext, i} * h_{fext, i})[n] + \sum_{j} (x_{next, j} * h_{next, j})[n] + v[n]$$

Where $*$ denotes convolution, $x$ denotes the transmitted signals, and $v[n]$ is AWGN.

## 2. Python Implementation

Because a DSP pipeline is traditionally linear ($A \rightarrow B \rightarrow C$), injecting multiple external aggressor signals requires passing them as optional arguments during the `process()` call. If no explicit aggressor signals are provided, the block can optionally auto-generate worst-case uncorrelated random data (dummy aggressors) to simulate the crosstalk noise floor.

```python
import numpy as np
from scipy.signal import lfilter
from typing import Any, Optional
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class CrosstalkChannelBlock(DSPBlock):
    """
    Simulates a high-speed link with THRU, NEXT, and FEXT paths.
    """
    def __init__(self, 
                 thru_taps: np.ndarray, 
                 next_taps: Optional[np.ndarray] = None, 
                 fext_taps: Optional[np.ndarray] = None) -> None:
        """
        Initializes the channel with time-domain impulse responses for each path.
        """
        super().__init__(
            thru_taps=np.asarray(thru_taps),
            next_taps=np.asarray(next_taps) if next_taps is not None else None,
            fext_taps=np.asarray(fext_taps) if fext_taps is not None else None
        )
        self.reset()

    def reset(self) -> None:
        """Initialize separate delay lines for THRU, NEXT, and FEXT filters."""
        self.state_thru = np.zeros(len(self._params['thru_taps']) - 1, dtype=np.complex128)
        
        if self._params['next_taps'] is not None:
            self.state_next = np.zeros(len(self._params['next_taps']) - 1, dtype=np.complex128)
            
        if self._params['fext_taps'] is not None:
            self.state_fext = np.zeros(len(self._params['fext_taps']) - 1, dtype=np.complex128)

    def _generate_dummy_aggressor(self, length: int) -> np.ndarray:
        """Generates a random QPSK/PAM4-like signal if no explicit aggressor is provided."""
        # For baseband RF, random complex; for SerDes, random PAM (-1, 1, etc.)
        # Here we use standard normal distribution as a generic proxy
        return (np.random.randn(length) + 1j * np.random.randn(length)) / np.sqrt(2)

    def process(self, 
                input_buf: SignalBuffer, 
                next_signal: Optional[SignalBuffer] = None, 
                fext_signal: Optional[SignalBuffer] = None,
                **kwargs: Any) -> SignalBuffer:
        
        N = input_buf.active_length
        
        # 1. Process THRU (Main Signal)
        y_total, self.state_thru = lfilter(
            b=self._params['thru_taps'], a=[1.0], 
            x=input_buf.data, zi=self.state_thru
        )
        
        # 2. Process NEXT (Near-End Crosstalk)
        if self._params['next_taps'] is not None:
            x_next = next_signal.data if next_signal else self._generate_dummy_aggressor(N)
            y_next, self.state_next = lfilter(
                b=self._params['next_taps'], a=[1.0], 
                x=x_next, zi=self.state_next
            )
            y_total += y_next
            
        # 3. Process FEXT (Far-End Crosstalk)
        if self._params['fext_taps'] is not None:
            x_fext = fext_signal.data if fext_signal else self._generate_dummy_aggressor(N)
            y_fext, self.state_fext = lfilter(
                b=self._params['fext_taps'], a=[1.0], 
                x=x_fext, zi=self.state_fext
            )
            y_total += y_fext
            
        return input_buf.clone_metadata(y_total)
```

## 3. Extracting Taps from S-Parameter Files

In practice, you will rarely hardcode these taps. Instead, you will load a multi-port Touchstone file (e.g., `.s8p` for a 2-lane differential link) using `scikit-rf`, similar to the `SParameterChannel` created in Phase 2.

### Port Assignment Example (8-Port Differential)
Assuming ports 1, 3 are Lane 0 (Tx/Rx) and ports 2, 4 are Lane 1 (Tx/Rx).
Converted to Mixed-Mode (Differential ports):
*   **THRU ($S_{dd21}$):** Differential transmission on Lane 0.
*   **NEXT ($S_{dd12}$ or similar):** Tx Lane 1 coupling into Rx Lane 1 at the same physical end.
*   **FEXT ($S_{dd41}$):** Tx Lane 0 coupling into Rx Lane 1 at the far end.

### Transformation Strategy
1. Load the `.s8p` file via `skrf.Network`.
2. Convert single-ended ports to mixed-mode (differential).
3. Extract the $S_{dd}$ magnitude and phase for THRU, NEXT, and FEXT frequency responses.
4. Perform the Inverse Fast Fourier Transform (IFFT) as detailed in the `s4p_channel_design.md` to yield `thru_taps`, `next_taps`, and `fext_taps`.
5. Inject these generated arrays directly into the `CrosstalkChannelBlock`.

## 4. Pipeline Integration

This block seamlessly replaces the basic `SParameterChannel` when running advanced Signal Integrity (SI) simulations.

```python
# Assuming taps were generated from IFFT of an .s8p file
crosstalk_channel = CrosstalkChannelBlock(
    thru_taps=thru_ir, 
    next_taps=next_ir, 
    fext_taps=fext_ir
)

system = DSPPipeline([crosstalk_channel, AWGNChannel(snr_db=30)])

# Simulate with auto-generated worst-case crosstalk
rx_signal = system.process(tx_signal)

# OR Simulate with explicit synchronous lanes
# rx_signal = system.process(tx_lane_0, next_signal=tx_lane_1_near, fext_signal=tx_lane_1_far)
```