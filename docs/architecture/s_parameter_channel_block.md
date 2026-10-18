# 📉 S-Parameter Channel Modeling (.s4p)

In high-speed links (like SerDes, PCIe, or Ethernet) and RF systems, the physical channel (PCB traces, cables, connectors) is often measured or simulated and exported as a Touchstone file, commonly an `.s4p` (4-port S-parameter) file.

To use an `.s4p` file in our time-domain `DSPPipeline`, we must convert its frequency-domain response into a time-domain **Impulse Response (FIR filter taps)**.

## 1. The Strategy: Frequency to Time Domain

1. **Parse the File:** Use the `scikit-rf` (`skrf`) Python library to load the `.s4p` file.
2. **Extract Transfer Function:** For differential signaling (common in `.s4p`), we typically extract the differential-to-differential forward transmission coefficient ($S_{dd21}$).
3. **IFFT Transformation:** We interpolate the frequency data to match our simulation's FFT grid (from DC up to the Nyquist frequency) and apply an Inverse Fast Fourier Transform (IFFT) to get the time-domain impulse response ($h[n]$).
4. **FIR Filtering:** We use the resulting $h[n]$ as the taps for a standard `DSPBlock` using `scipy.signal.lfilter`, just like the Multipath channel.

## 2. Dependencies

You will need to add `scikit-rf` to your project dependencies:
```bash
pip install scikit-rf
```

## 3. Python Implementation: `SParameterChannel`

Here is the implementation of an `.s4p` channel block.

```python
import numpy as np
import skrf as rf
from scipy.signal import lfilter
from typing import Any
from dsp_block_api import DSPBlock

class SParameterChannel(DSPBlock):
    """
    Simulates a physical channel based on an .s4p Touchstone file.
    It extracts the differential insertion loss (Sdd21) and converts it to an FIR filter.
    """
    def __init__(self, s4p_filepath: str, sample_rate_hz: float, target_taps: int = 1024) -> None:
        super().__init__(
            s4p_filepath=s4p_filepath, 
            sample_rate_hz=sample_rate_hz,
            target_taps=target_taps
        )
        self._load_and_convert_s4p()
        self.reset()
        
    def _load_and_convert_s4p(self) -> None:
        """Reads the .s4p file and generates the time-domain impulse response."""
        filepath = self._params['s4p_filepath']
        fs = self._params['sample_rate_hz']
        n_taps = self._params['target_taps']
        
        # 1. Load Touchstone file
        network = rf.Network(filepath)
        
        # 2. Convert 4-port single-ended to 2-port mixed-mode (Differential & Common mode)
        # Assuming standard port assignment: 1,3 are input pair; 2,4 are output pair
        network.se2gmm(p=2) 
        
        # Extract Sdd21 (Differential Insertion Loss)
        # In scikit-rf mixed mode: port 0 is diff 1, port 1 is diff 2
        sdd21 = network.s[:, 1, 0] 
        freqs = network.f
        
        # 3. Interpolate to an even frequency grid from DC to Nyquist (fs/2)
        target_freqs = np.linspace(0, fs / 2, n_taps // 2 + 1)
        
        # Complex interpolation
        sdd21_mag = np.interp(target_freqs, freqs, np.abs(sdd21))
        sdd21_phase = np.interp(target_freqs, freqs, np.unwrap(np.angle(sdd21)))
        
        # Extrapolate DC (0 Hz) explicitly if not in original data to avoid offsets
        sdd21_phase[0] = 0.0 
        
        H_target = sdd21_mag * np.exp(1j * sdd21_phase)
        
        # 4. Perform IFFT. irfft expects a spectrum from DC to Nyquist and returns real taps
        impulse_response = np.fft.irfft(H_target, n=n_taps)
        
        # Apply a window (e.g., Hamming) to reduce truncation ringing
        window = np.hamming(n_taps)
        self.taps = impulse_response * window
        
    def reset(self) -> None:
        """Clear the FIR filter delay line."""
        # lfilter expects state array of length max(len(a), len(b)) - 1
        self.filter_state = np.zeros(len(self.taps) - 1, dtype=np.float64)

    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the channel impulse response to the signal."""
        # Convolution with state preservation for continuous streaming
        output_data, self.filter_state = lfilter(
            b=self.taps, 
            a=[1.0], 
            x=input_data, 
            zi=self.filter_state
        )
        return output_data
```

## 4. Usage in the Pipeline

This block integrates perfectly with the rest of the framework. You can load an S-parameter file and cascade it with AWGN.

```python
# Create the channel blocks
pcb_channel = SParameterChannel(
    s4p_filepath="data/pcb_trace_10inch.s4p",
    sample_rate_hz=56e9, # 56 Gbaud
    target_taps=2048
)
noise = AWGNChannel(snr_db=25.0)

# Build the pipeline
tx_to_rx_link = DSPPipeline([pcb_channel, noise])

# Simulate
received_signal = tx_to_rx_link.process(transmitted_signal)
```

## 5. Advanced Considerations for C++ Migration
In the future, when migrating this to C++:
* The `.s4p` parsing and IFFT conversion should still happen in Python during initialization (since it's a one-time calculation).
* Only the calculated `self.taps` array needs to be passed down to the C++ core.
* The C++ core will then just run a highly optimized FIR convolution engine, completely unaware of where the taps came from!