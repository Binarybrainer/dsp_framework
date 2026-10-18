# 📡 MIMO Processing Specification

Multiple-Input Multiple-Output (MIMO) technology uses multiple antennas at the transmitter and receiver to multiply data throughput and increase link reliability. 

This document outlines the implementation of a basic flat-fading MIMO channel (the $H$ matrix) and the two most fundamental linear equalizers: **Zero-Forcing (ZF)** and **Minimum Mean Square Error (MMSE)**.

## 1. Multi-Antenna Data Representation

To support MIMO, we assume the `SignalBuffer.data` attribute holds a 2D NumPy array of shape `(N_samples, N_antennas)`. 

* A $2 \times 2$ MIMO system will have `input_buf.data.shape == (N_samples, 2)`.
* NumPy's matrix multiplication (`@`) handles this multi-dimensional streaming natively and efficiently.

## 2. MIMO Channel Modeling (The $H$ Matrix)

The received signal $\mathbf{y}$ in a flat-fading MIMO system is mathematically modeled as:

$$
\mathbf{y}[n] = \mathbf{H} \mathbf{x}[n] + \mathbf{n}[n]
$$

Where:
* $\mathbf{x}[n]$ is the transmitted vector of size $N_{tx} \times 1$.
* $\mathbf{y}[n]$ is the received vector of size $N_{rx} \times 1$.
* $\mathbf{H}$ is the complex channel matrix of size $N_{rx} \times N_{tx}$.
* $\mathbf{n}[n]$ is the AWGN noise vector.

### Python Implementation (`MIMOFlatFadingChannel`)

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

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

```

## 3. Linear Equalizers: Zero-Forcing (ZF) and MMSE

At the receiver, the goal is to recover the transmitted vector $\mathbf{x}$ from the received vector $\mathbf{y}$. We apply a weight matrix $\mathbf{W}$ such that $\hat{\mathbf{x}} = \mathbf{W} \mathbf{y}$.

### 3.1. Zero-Forcing (ZF) Equalizer
The ZF equalizer completely inverts the channel to eliminate inter-antenna interference. 
$$
\mathbf{W}_{ZF} = (\mathbf{H}^H \mathbf{H})^{-1} \mathbf{H}^H = \mathbf{H}^{\dagger}
$$
*Note: $\mathbf{H}^{\dagger}$ is the Moore-Penrose pseudo-inverse. While ZF eliminates interference, it severely amplifies noise when channel matrix eigenvalues are small.*

### 3.2. Minimum Mean Square Error (MMSE) Equalizer
The MMSE equalizer balances channel inversion with noise amplification by taking the Signal-to-Noise Ratio (SNR) into account.
$$
\mathbf{W}_{MMSE} = (\mathbf{H}^H \mathbf{H} + \frac{1}{SNR} \mathbf{I})^{-1} \mathbf{H}^H
$$

### Python Implementation (`MIMOLinearEqualizer`)

Assuming the receiver has perfect Channel State Information (CSI) - meaning it knows $\mathbf{H}$ perfectly.

```python
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

```

## 4. Pipeline Integration (2x2 MIMO Example)

Here is how you simulate a full $2 \times 2$ spatial multiplexing MIMO link.

```python
# 1. Generate multi-antenna data (2 streams of QPSK symbols)
N_samples = 1000
tx_data_stream0 = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], N_samples) / np.sqrt(2)
tx_data_stream1 = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], N_samples) / np.sqrt(2)

# Stack into (1000, 2) array
tx_matrix = np.column_stack((tx_data_stream0, tx_data_stream1))
tx_buf = SignalBuffer(tx_matrix, sample_rate=1e6)

# 2. Channel & Impairments
mimo_channel = MIMOFlatFadingChannel(num_tx=2, num_rx=2)
# Re-using our existing AWGN block (it naturally handles N-dimensional arrays!)
noise = AWGNChannel(snr_db=20.0) 

# 3. Simulate Physical Link
rx_buf = noise.process(mimo_channel.process(tx_buf))

# 4. Equalization at Receiver (Assuming perfect CSI)
snr_linear = 10**(20.0 / 10.0)
mmse_eq = MIMOLinearEqualizer(H_est=mimo_channel.H, snr_linear=snr_linear, method='mmse')

# 5. Recover transmitted streams
recovered_buf = mmse_eq.process(rx_buf)

# Plot Constellation of recovered Stream 0
# (You can modify the plot_constellation utility to select a specific antenna index: data[:, 0])

```