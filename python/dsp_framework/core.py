from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

class DSPBlock(ABC):
    """
    Abstract Base Class for all Digital Signal Processing blocks in the framework.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the DSP block with arbitrary parameters.
        Child classes should call super().__init__(**kwargs) to register parameters.
        """
        self._params: Dict[str, Any] = kwargs
        self.reset()

    @abstractmethod
    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Core signal processing routine.
        
        Args:
            input_data (np.ndarray): The input signal array.
            **kwargs: Optional block-specific processing flags.
            
        Returns:
            np.ndarray: The processed output signal array.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state/memory of the DSP block.
        This must be implemented by child classes to clear delay lines, 
        accumulators, or state machine variables.
        """
        pass

    def update_params(self, **kwargs: Any) -> None:
        """
        Update the configuration parameters of the block on the fly.
        
        Args:
            **kwargs: Key-value pairs of parameters to update.
            
        Raises:
            KeyError: If a parameter being updated does not exist in the block.
        """
        for key, value in kwargs.items():
            if key in self._params:
                self._params[key] = value
                # Child classes can override this method to trigger specific 
                # recalculations (e.g., recalculating filter coefficients) 
                # when a parameter changes.
            else:
                raise KeyError(f"Parameter '{key}' is not a valid parameter for {self.__class__.__name__}.")

    @property
    def params(self) -> Dict[str, Any]:
        """Get the current parameters of the block."""
        return self._params


class DSPPipeline(DSPBlock):
    """
    A composite DSP block that chains multiple DSPBlocks together sequentially.
    """

    def __init__(self, blocks: list[DSPBlock] = None, **kwargs: Any) -> None:
        """
        Initialize the pipeline with an optional list of blocks.
        """
        super().__init__(**kwargs)
        self.blocks: list[DSPBlock] = blocks if blocks is not None else []

    def add_block(self, block: DSPBlock) -> None:
        """
        Append a new DSPBlock to the end of the pipeline.
        """
        if not isinstance(block, DSPBlock):
            raise TypeError("Only instances of DSPBlock can be added to the pipeline.")
        self.blocks.append(block)

    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Pass the input data through all blocks in the pipeline sequentially.
        
        Args:
            input_data (np.ndarray): The initial input signal.
            
        Returns:
            np.ndarray: The final processed signal.
        """
        current_data = input_data
        for block in self.blocks:
            # The output of block N becomes the input of block N+1
            current_data = block.process(current_data)
        
        return current_data

    def reset(self) -> None:
        """
        Reset the internal state of every block in the pipeline.
        """
        for block in getattr(self, 'blocks', []):
            block.reset()
            
    def update_params(self, **kwargs: Any) -> None:
        """
        Overrides the default update_params to broadcast parameter updates 
        to all child blocks. Note: This requires careful naming to avoid collisions.
        """
        super().update_params(**kwargs)
        for block in self.blocks:
            # Only update parameters that the specific block recognizes
            block_kwargs = {k: v for k, v in kwargs.items() if k in block.params}
            if block_kwargs:
                block.update_params(**block_kwargs)

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


from typing import Any
import numpy as np

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

import numpy as np
from typing import Any

class SymbolMapperBlock(DSPBlock):
    """
    Maps integer symbol indices (0 to M-1) to complex constellation points.
    Supports BPSK, QPSK, and Square M-QAM (16, 64, 256...).
    """
    def __init__(self, modulation_type: str = 'QPSK', M: int = 4) -> None:
        super().__init__(modulation_type=modulation_type.upper(), M=M)
        self._generate_constellation()
        self.reset()
        
    def reset(self) -> None:
        pass # Mapping is memoryless

    def _generate_constellation(self) -> None:
        """Generates the power-normalized Gray-coded constellation lookup table."""
        mod_type = self._params['modulation_type']
        M = self._params['M']
        
        if mod_type == 'BPSK' or (mod_type == 'QAM' and M == 2):
            self.constellation = np.array([-1.0 + 0j, 1.0 + 0j])
            
        elif mod_type == 'QPSK' or (mod_type == 'QAM' and M == 4):
            # Standard Gray-coded QPSK (pi/4 offset)
            self.constellation = np.array([
                -1.0 - 1j, -1.0 + 1j, 1.0 - 1j, 1.0 + 1j
            ]) / np.sqrt(2)
            
        elif mod_type == 'QAM' and np.sqrt(M).is_integer():
            # Generate Square QAM
            k = int(np.sqrt(M))
            # Generate PAM levels: e.g., for 16-QAM, k=4, PAM = [-3, -1, 1, 3]
            pam_levels = 2 * np.arange(k) - (k - 1)
            
            # Create grid
            real_grid, imag_grid = np.meshgrid(pam_levels, pam_levels)
            const = (real_grid + 1j * imag_grid).flatten()
            
            # Simple binary to Gray mapping for the lookup table index
            # (In a production system, an exact bit-mapped Gray code dictionary is used)
            gray_indices = self._binary_to_gray(np.arange(M))
            self.constellation = np.zeros(M, dtype=np.complex128)
            self.constellation[gray_indices] = const
            
            # Normalize to unit average power
            avg_power = np.mean(np.abs(self.constellation)**2)
            self.constellation /= np.sqrt(avg_power)
        else:
            raise ValueError(f"Unsupported modulation: {mod_type} with M={M}")

    def _binary_to_gray(self, n: np.ndarray) -> np.ndarray:
        """Helper to convert binary indices to Gray code indices."""
        return n ^ (n >> 1)

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        """
        Input: SignalBuffer of integer symbols.
        Output: SignalBuffer of complex IQ symbols.
        """
        # NumPy advanced indexing acts as a blazing fast lookup table
        mapped_symbols = self.constellation[input_buf.data.astype(int)]
        
        # We output a new buffer with the same time/fs metadata
        return input_buf.clone_metadata(mapped_symbols)

class SymbolDemapperBlock(DSPBlock):
    """
    Demaps noisy complex symbols back to integer symbols (Hard) 
    or Bit Log-Likelihood Ratios (Soft).
    """
    def __init__(self, constellation: np.ndarray, decision_type: str = 'hard', noise_var: float = 1.0) -> None:
        """
        constellation: The exact lookup table used by the Mapper.
        decision_type: 'hard' or 'soft'.
        noise_var: Noise variance (N0), required for accurate Soft LLR calculation.
        """
        super().__init__(constellation=np.asarray(constellation), 
                         decision_type=decision_type.lower(), 
                         noise_var=noise_var)
        self.reset()
        
    def reset(self) -> None:
        pass

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        decision = self._params['decision_type']
        const = self._params['constellation']
        data = input_buf.data
        
        if decision == 'hard':
            # 1. Expand dimensions for broadcasting: 
            # data becomes (N, 1), const becomes (1, M)
            # Distance matrix becomes (N, M)
            dist_sq = np.abs(data[:, np.newaxis] - const[np.newaxis, :])**2
            
            # 2. Find the index (symbol) with the minimum distance
            hard_symbols = np.argmin(dist_sq, axis=1)
            
            return input_buf.clone_metadata(hard_symbols)
            
        elif decision == 'soft':
            # Max-Log-MAP approximation for LLRs
            # LLR(bit i) ~ (min_dist_to_bit_0 - min_dist_to_bit_1) / noise_var
            
            M = len(const)
            num_bits = int(np.log2(M))
            dist_sq = np.abs(data[:, np.newaxis] - const[np.newaxis, :])**2
            
            N = len(data)
            llrs = np.zeros((N, num_bits), dtype=np.float64)
            
            for b in range(num_bits):
                # Identify which constellation points have bit 'b' as 0 and as 1
                # (Requires knowledge of the exact bit-mapping of the constellation)
                mask_1 = (np.arange(M) & (1 << b)) > 0
                mask_0 = ~mask_1
                
                # Minimum distance to a '1' point and '0' point
                min_d1 = np.min(dist_sq[:, mask_1], axis=1)
                min_d0 = np.min(dist_sq[:, mask_0], axis=1)
                
                # LLR > 0 means more likely 0. LLR < 0 means more likely 1.
                llrs[:, b] = (min_d1 - min_d0) / self._params['noise_var']
                
            return input_buf.clone_metadata(llrs.flatten())
            
        else:
            raise ValueError(f"Unknown decision type: {decision}")

from scipy.signal import lfilter, lfiltic

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


class PLLBlock(DSPBlock):
    """
    A digital Phase-Locked Loop (PLL) for carrier tracking and phase synchronization.
    Includes a Phase Detector, a PI Loop Filter, and an NCO.
    """
    def __init__(self, kp: float = 0.1, ki: float = 0.01, center_freq: float = 0.0) -> None:
        """
        Args:
            kp: Proportional gain for the loop filter.
            ki: Integral gain for the loop filter.
            center_freq: Nominal center frequency offset.
        """
        super().__init__(kp=kp, ki=ki, center_freq=center_freq)
        self.reset()
        
    def reset(self) -> None:
        """Reset the PLL states for continuous streaming."""
        self.phase_state = 0.0
        self.integral_state = 0.0

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        kp = self._params['kp']
        ki = self._params['ki']
        w0 = self._params['center_freq']
        
        N = len(data)
        out_data = np.zeros_like(data, dtype=np.complex128)
        
        # We must process sample-by-sample due to the feedback loop
        for i in range(N):
            # 1. NCO Output
            nco_out = np.exp(1j * self.phase_state)
            
            # 2. Phase Detector (PD)
            # Calculate phase error using the complex conjugate product
            # error = Im(data * conj(nco_out))
            phase_error = np.imag(data[i] * np.conj(nco_out))
            
            # 3. Loop Filter (PI)
            self.integral_state += ki * phase_error
            control_voltage = kp * phase_error + self.integral_state
            
            # 4. Update NCO Phase for the next sample
            self.phase_state += w0 + control_voltage
            
            # Wrap phase to stay within [-pi, pi]
            self.phase_state = (self.phase_state + np.pi) % (2 * np.pi) - np.pi
            
            # Output the synchronized / de-rotated signal
            out_data[i] = data[i] * np.conj(nco_out)
            
        return input_buf.clone_metadata(out_data)

class BangBangCDRBlock(DSPBlock):
    """
    Clock and Data Recovery using an Alexander (Bang-Bang) Phase Detector 
    and an oversampled Phase Interpolator (PI).
    """
    def __init__(self, sps: int, kp: float = 0.05, ki: float = 0.005) -> None:
        """
        sps: Samples per symbol of the incoming oversampled data.
        kp: Proportional loop gain.
        ki: Integral loop gain.
        """
        super().__init__(sps=sps, kp=kp, ki=ki)
        self.reset()
        
    def reset(self) -> None:
        # Loop Filter states
        self.lf_integral = 0.0
        
        # PI State: Tracks the fractional offset in terms of sub-samples
        # Start sampling roughly in the middle of the oversampled eye
        self.pi_phase = self._params['sps'] / 2.0 
        
        # Memory of the last recovered data symbol
        self.last_data_sample = 0.0

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        sps = self._params['sps']
        kp = self._params['kp']
        ki = self._params['ki']
        
        # We will extract one symbol per 'sps' roughly.
        # Pre-allocate output assuming input length / sps
        estimated_symbols = len(data) // sps
        recovered_data = np.zeros(estimated_symbols, dtype=np.complex128)
        
        # For tracking Phase Interpolator movement (useful for debugging plots)
        pi_track = np.zeros(estimated_symbols, dtype=np.float64)
        
        out_idx = 0
        read_ptr = 0.0 # Points to the start of the current UI (Unit Interval)
        
        while int(read_ptr + sps) < len(data) and out_idx < estimated_symbols:
            # 1. Phase Interpolator (PI) extracts the Edge and Data samples
            # Edge is nominally at 0 phase, Data is nominally at PI Phase (center)
            
            # Sub-sample indices
            edge_idx = int(read_ptr)
            data_idx = int(read_ptr + self.pi_phase)
            
            E_n = np.sign(data[edge_idx].real) # Hard slice edge
            D_n = np.sign(data[data_idx].real) # Hard slice data
            D_prev = self.last_data_sample
            
            # 2. Alexander Bang-Bang Phase Detector (BBPD)
            pd_out = 0.0
            if D_prev != D_n:
                if E_n == D_prev:
                    pd_out = 1.0  # Clock is Early
                else:
                    pd_out = -1.0 # Clock is Late
                    
            # 3. Loop Filter (PI Controller)
            self.lf_integral += ki * pd_out
            pi_code = kp * pd_out + self.lf_integral
            
            # 4. Phase Interpolator Update
            # Adjust the phase for the NEXT symbol. 
            # We shift the start of the next UI by the nominal sps + the PI adjustment
            read_ptr += sps + pi_code
            
            # Save data
            recovered_data[out_idx] = data[data_idx]
            pi_track[out_idx] = pi_code
            self.last_data_sample = D_n
            out_idx += 1
            
        # Create output buffer (Note: the sample rate is now decimated by roughly 'sps')
        out_fs = input_buf.sample_rate / sps
        
        # We can store the pi_track in kwargs or a custom attribute if needed for debugging
        self.last_pi_track = pi_track[:out_idx]
        
        return SignalBuffer(
            data=recovered_data[:out_idx], 
            sample_rate=out_fs, 
            timestamp=input_buf.timestamp
        )

import plotly.graph_objects as go
import numpy as np

class EyeDiagramPlotter:
    
    @staticmethod
    def plot_eye(buffer: SignalBuffer, sps: int, num_uis: int = 2, max_traces: int = 500, title: str = "Eye Diagram") -> go.Figure:
        """
        Generates an interactive Eye Diagram using Plotly WebGL.
        
        Args:
            buffer: The heavily oversampled SignalBuffer (BEFORE the CDR).
            sps: Samples per symbol (Unit Interval).
            num_uis: How many symbols wide the eye diagram should be (usually 2 or 3).
            max_traces: Limit the number of overlaid traces to prevent browser lag.
        """
        data = buffer.data.real # Eye diagrams are typically plotted for I and Q separately
        
        # Calculate length of one trace
        trace_length = sps * num_uis
        
        # Determine how many traces we can extract
        total_possible_traces = len(data) // trace_length
        num_traces = min(total_possible_traces, max_traces)
        
        # Reshape data into a 2D array: (num_traces, trace_length)
        # This acts as our "Oscilloscope Trigger" folding
        folded_data = np.reshape(
            data[:num_traces * trace_length], 
            (num_traces, trace_length)
        )
        
        # X-axis represented in Unit Intervals (UI)
        x_axis_ui = np.linspace(0, num_uis, trace_length)
        
        fig = go.Figure()
        
        # Add multiple traces to the plot using Scattergl for performance
        for i in range(num_traces):
            fig.add_trace(go.Scattergl(
                x=x_axis_ui,
                y=folded_data[i, :],
                mode='lines',
                line=dict(color='rgba(0, 50, 255, 0.1)', width=1), # Low opacity blue
                showlegend=False,
                hoverinfo='skip' # Skip hover on individual traces to save CPU
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title="Time (Unit Intervals - UI)",
            yaxis_title="Amplitude",
            width=800, height=500
        )
        
        return fig


class MIMOFlatFadingChannel(DSPBlock):
    """
    Simulates a static, frequency-flat Rayleigh fading MIMO channel.
    """
    def __init__(self, num_tx: int = 2, num_rx: int = 2) -> None:
        super().__init__(num_tx=num_tx, num_rx=num_rx)
        self._generate_h_matrix()
        self.reset()
        
    def reset(self) -> None:
        pass

    def _generate_h_matrix(self) -> None:
        """Generates a random complex Rayleigh fading matrix."""
        num_tx = self._params['num_tx']
        num_rx = self._params['num_rx']
        
        # H is sized (N_rx, N_tx). Elements are complex Gaussian with unit variance.
        self.H = (np.random.randn(num_rx, num_tx) + 1j * np.random.randn(num_rx, num_tx)) / np.sqrt(2)

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        """
        Applies the H matrix to the transmitted data.
        input_buf.data shape: (N_samples, N_tx)
        """
        X = input_buf.data
        
        # Apply channel: Y = X @ H^T
        # Shape output: (N_samples, N_rx)
        Y = X @ self.H.T
        
        return input_buf.clone_metadata(Y)

class MIMOLinearEqualizer(DSPBlock):
    """
    Applies ZF or MMSE linear equalization to a received MIMO signal.
    Assumes perfect Channel State Information (CSI) is passed to the block.
    """
    def __init__(self, H_est: np.ndarray, snr_linear: float = 100.0, method: str = 'mmse') -> None:
        """
        H_est: The estimated channel matrix (N_rx, N_tx).
        snr_linear: Signal to Noise Ratio in linear scale (required for MMSE).
        method: 'zf' or 'mmse'.
        """
        super().__init__(H_est=H_est, snr_linear=snr_linear, method=method.lower())
        self._calculate_weights()
        self.reset()
        
    def reset(self) -> None:
        pass

    def _calculate_weights(self) -> None:
        H = self._params['H_est']
        snr = self._params['snr_linear']
        method = self._params['method']
        
        num_rx, num_tx = H.shape
        H_hermitian = H.conj().T
        
        if method == 'zf':
            # Pseudo-inverse of H
            self.W = np.linalg.pinv(H)
            
        elif method == 'mmse':
            # MMSE formula: (H^H * H + (1/SNR) * I)^-1 * H^H
            identity = np.eye(num_tx)
            inverse_term = np.linalg.inv((H_hermitian @ H) + (1.0 / snr) * identity)
            self.W = inverse_term @ H_hermitian
            
        else:
            raise ValueError(f"Unknown equalization method: {method}")

    def update_params(self, **kwargs: Any) -> None:
        super().update_params(**kwargs)
        # Recalculate weight matrix if H_est or SNR changes
        if 'H_est' in kwargs or 'snr_linear' in kwargs or 'method' in kwargs:
            self._calculate_weights()

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        """
        Applies the equalization matrix W to the received data.
        input_buf.data shape: (N_samples, N_rx)
        """
        Y = input_buf.data
        
        # Apply equalization: X_hat = Y @ W^T
        # Shape output: (N_samples, N_tx)
        X_hat = Y @ self.W.T
        
        return input_buf.clone_metadata(X_hat)