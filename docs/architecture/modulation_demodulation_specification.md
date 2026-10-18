# 🎛️ Modulation & Demodulation Design

The core of any digital communication system is mapping discrete digital data (bits or symbol indices) into analog waveforms (complex IQ symbols) for transmission, and recovering that data at the receiver.

This document specifies the **Symbol Mapper** and **Symbol Demapper** blocks. Because mapping is a memoryless, element-wise operation, we will heavily rely on NumPy vectorization to achieve near-C++ speeds natively in Python.

## 1. Symbol Mapper (BPSK, QPSK, M-QAM)

The `SymbolMapperBlock` takes an input `SignalBuffer` containing integer symbol indices (e.g., $0$ to $M-1$) and maps them to a complex constellation using Gray coding. The constellation is automatically normalized to have an average unit power ($E_s = 1$).

### Python Implementation

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

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
```

## 2. Symbol Demapper (Hard vs. Soft Decision)

The Demapper translates received noisy complex symbols back into digital data. 
* **Hard Decision:** Outputs definitive integer symbols (e.g., this is a `3`). It uses Minimum Euclidean Distance.
* **Soft Decision:** Outputs Log-Likelihood Ratios (LLRs). Instead of saying "this bit is 1", it says "I am 80% confident this bit is 1". This is mandatory for modern Forward Error Correction (FEC) like LDPC or Turbo Codes.

### Python Implementation

We use NumPy broadcasting to calculate distances across the entire array simultaneously, keeping the execution entirely in C.

```python
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
```

## 3. Pipeline Integration

Here is how you chain the data generation, mapping, channel, and demapping.

```python
# 1. Setup
M = 16 # 16-QAM
mapper = SymbolMapperBlock(modulation_type='QAM', M=M)
demapper = SymbolDemapperBlock(constellation=mapper.constellation, decision_type='hard')
channel = AWGNChannel(snr_db=20.0)

# 2. Generate random integer symbols
tx_symbols = np.random.randint(0, M, 1000)
tx_buf = SignalBuffer(tx_symbols, sample_rate=1e6)

# 3. Simulate Pipeline
# Map to IQ -> Add Noise -> Demap to Symbols
iq_buf = mapper.process(tx_buf)
noisy_iq_buf = channel.process(iq_buf)
rx_symbols_buf = demapper.process(noisy_iq_buf)

# 4. Calculate Symbol Error Rate (SER)
errors = np.sum(tx_buf.data != rx_symbols_buf.data)
print(f"Symbol Error Rate: {errors / len(tx_symbols)}")
```