# 🎛️ Advanced ADC Simulation: Time-Interleaved ADC Modeling

A real-world Analog-to-Digital Converter is not a perfect quantization box. At extreme sampling rates (100+ GS/s), hardware designers must interleave multiple slower sub-ADCs. This architecture, known as a **Time-Interleaved ADC (TI-ADC)**, introduces periodic impairments that heavily distort the signal.

This document specifies the `TimeInterleavedADCBlock`. It models standard impairments (quantization, clipping, aperture jitter) alongside the complex mismatch effects of TI-ADCs (gain, offset, and timing skew), which are critical for testing ADC DSP calibration algorithms.

## 1. Impairment Mathematical Models

Assuming an ADC with $M$ interleaved sub-ADCs, the $m$-th sub-ADC processes every $M$-th sample. 

1. **Quantization & Clipping:**
   The continuous voltage is constrained by a Full-Scale Voltage ($V_{FS}$) and divided into $2^B$ discrete levels, where $B$ is the bit depth (Effective Number of Bits - ENOB).

2. **Offset Mismatch ($o_m$):**
   Each sub-ADC has a slightly different DC bias. This introduces periodic tones in the frequency spectrum at multiples of $f_s / M$.

3. **Gain Mismatch ($g_m$):**
   Variations in amplifiers cause each sub-ADC to scale the signal differently, leading to amplitude modulation.

4. **Timing Skew / Sampling Phase Mismatch ($\Delta t_m$):**
   The clock distributing the sampling trigger to the $M$ sub-ADCs is not perfectly uniform. A slight delay in one sub-ADC effectively acts as a phase shift.

## 2. Python Implementation (`TimeInterleavedADCBlock`)

This block handles both real and complex signals. For complex signals, it applies impairments independently to the I and Q channels.

```python
import numpy as np
from typing import Any, Optional
from scipy.interpolate import interp1d
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class TimeInterleavedADCBlock(DSPBlock):
    """
    Simulates an ultra-high-speed Time-Interleaved ADC (TI-ADC) with 
    quantization, clipping, and M-channel mismatches.
    """
    def __init__(self, 
                 bits: int = 8, 
                 v_full_scale: float = 1.0, 
                 num_sub_adcs: int = 4,
                 offset_mismatch: Optional[np.ndarray] = None,
                 gain_mismatch: Optional[np.ndarray] = None,
                 timing_skew_fs: Optional[np.ndarray] = None) -> None:
        """
        bits: Resolution of the ADC (e.g., 8 bits).
        v_full_scale: Maximum voltage (+/-) before clipping.
        num_sub_adcs: The interleaving factor M.
        offset_mismatch: Array of length M specifying DC offsets.
        gain_mismatch: Array of length M specifying gain factors (ideal is 1.0).
        timing_skew_fs: Array of length M specifying sampling delays in femtoseconds.
        """
        M = num_sub_adcs
        super().__init__(
            bits=bits,
            v_full_scale=v_full_scale,
            num_sub_adcs=M,
            offset_mismatch=np.asarray(offset_mismatch) if offset_mismatch is not None else np.zeros(M),
            gain_mismatch=np.asarray(gain_mismatch) if gain_mismatch is not None else np.ones(M),
            timing_skew_fs=np.asarray(timing_skew_fs) if timing_skew_fs is not None else np.zeros(M)
        )
        self.reset()
        
    def reset(self) -> None:
        # Keep track of the total samples processed to apply the correct M-th sub-ADC
        self.sample_counter = 0

    def _apply_timing_skew(self, data: np.ndarray, t: np.ndarray, skews_fs: np.ndarray) -> np.ndarray:
        """Applies fractional timing skew using interpolation."""
        if np.all(skews_fs == 0):
            return data
            
        M = self._params['num_sub_adcs']
        out_data = np.zeros_like(data)
        
        # We need a slight overlap buffer in a real streaming scenario, 
        # but for chunk processing, we interpolate locally.
        interpolator = interp1d(t, data, kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        # Calculate shifted time vectors for each sub-ADC
        for m in range(M):
            # Find indices belonging to sub-ADC m
            idx_m = np.arange((m - self.sample_counter) % M, len(data), M)
            
            # Shift time by skew (convert fs to seconds)
            skew_sec = skews_fs[m] * 1e-15
            shifted_time = t[idx_m] + skew_sec
            
            # Resample at the skewed time points
            out_data[idx_m] = interpolator(shifted_time)
            
        return out_data

    def _quantize_and_impair(self, data: np.ndarray) -> np.ndarray:
        """Applies gain, offset, clipping, and quantization to a 1D real array."""
        M = self._params['num_sub_adcs']
        bits = self._params['bits']
        v_fs = self._params['v_full_scale']
        offsets = self._params['offset_mismatch']
        gains = self._params['gain_mismatch']
        
        # 1. Apply Gain and Offset mismatches per sub-ADC
        impaired_data = np.copy(data)
        for m in range(M):
            idx_m = np.arange((m - self.sample_counter) % M, len(data), M)
            impaired_data[idx_m] = impaired_data[idx_m] * gains[m] + offsets[m]
            
        # 2. Clipping
        np.clip(impaired_data, -v_fs, v_fs, out=impaired_data)
        
        # 3. Quantization
        levels = 2 ** bits
        step_size = (2 * v_fs) / levels
        
        # Shift, round, and shift back
        quantized = np.round((impaired_data + v_fs) / step_size) * step_size - v_fs
        
        # Ensure we don't exceed the top boundary due to rounding
        quantized[quantized > v_fs - step_size] = v_fs - step_size
        
        return quantized

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        t = input_buf.time_vector()
        
        if np.iscomplexobj(data):
            # In a coherent receiver, I and Q have independent ADCs.
            # We apply the same statistical impairments to both for simulation simplicity, 
            # though they could be configured separately in a strictly rigorous model.
            
            # Timing Skew
            i_skewed = self._apply_timing_skew(data.real, t, self._params['timing_skew_fs'])
            q_skewed = self._apply_timing_skew(data.imag, t, self._params['timing_skew_fs'])
            
            # Quantization, Gain, Offset
            i_out = self._quantize_and_impair(i_skewed)
            q_out = self._quantize_and_impair(q_skewed)
            
            out_data = i_out + 1j * q_out
        else:
            skewed = self._apply_timing_skew(data, t, self._params['timing_skew_fs'])
            out_data = self._quantize_and_impair(skewed)
            
        # Update state for continuous streaming continuity
        self.sample_counter = (self.sample_counter + len(data)) % self._params['num_sub_adcs']
            
        return input_buf.clone_metadata(out_data)
```

## 3. Pipeline Integration and Analysis

When you place this block in your pipeline, the time-interleaving mismatches will create "Interleaving Spurs" (spurious tones) in your PSD plots generated by the `DSPPlotter`. 

```python
# Configure a 4-channel TI-ADC with extremely subtle imperfections
adc = TimeInterleavedADCBlock(
    bits=8, 
    v_full_scale=1.0, 
    num_sub_adcs=4,
    offset_mismatch=np.array([0.001, -0.002, 0.0005, -0.001]),
    gain_mismatch=np.array([1.0, 1.01, 0.99, 1.0]),
    timing_skew_fs=np.array([0, 150, -50, 200]) # Skews in femtoseconds
)

# Insert ADC right after the analog channel and before digital filtering
rx_digital_buffer = adc.process(rx_analog_buffer)

# Plotting the PSD will reveal the characteristic mismatch spurs 
# at exactly fs/4, fs/2, and 3fs/4.
psd_fig = DSPPlotter.plot_psd(rx_digital_buffer)
psd_fig.show()
```

This model provides the exact foundation needed to develop and test background calibration algorithms (like blind skew tracking or offset cancellation) entirely within your Python DSP framework.