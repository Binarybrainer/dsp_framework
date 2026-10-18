# 🧠 Adaptive Algorithms (LMS & Kalman Filters)

In dynamic environments, static filters are insufficient. The channel changes over time, requiring the receiver to continuously track and adapt. This document specifies the implementation of adaptive algorithms: the **Least Mean Squares (LMS)** algorithm for channel equalization and the **Kalman Filter** for optimal phase/state tracking.

Because both algorithms rely on sample-by-sample feedback loops (updating weights or state covariance at each step $n$), they will be prototyped as "Super-Blocks" in Python and are primary targets for C++ acceleration.

## 1. LMS Adaptive Equalizer

The LMS algorithm is widely used to reverse the effects of multipath fading and Inter-Symbol Interference (ISI). It is an FIR filter whose taps (weights) update themselves automatically to minimize the mean square error between the output and a desired reference signal (or a decision-directed estimate).

### Mathematical Model
1. **Filtering:** $y[n] = \mathbf{w}^H \mathbf{x}[n]$
2. **Error Calculation:** $e[n] = d[n] - y[n]$  (where $d[n]$ is the desired signal)
3. **Weight Update:** $\mathbf{w} \leftarrow \mathbf{w} + \mu \cdot e^*[n] \cdot \mathbf{x}[n]$

### Python Implementation (`LMS_EqualizerBlock`)

We implement a Decision-Directed LMS Equalizer, which uses its own closest symbol guess as the "desired" signal $d[n]$.

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class LMS_EqualizerBlock(DSPBlock):
    """
    Decision-Directed LMS Adaptive Equalizer.
    Continuously updates FIR weights to mitigate Intersymbol Interference (ISI).
    """
    def __init__(self, num_taps: int = 11, mu: float = 0.01) -> None:
        """
        num_taps: Length of the adaptive FIR filter.
        mu: Step size (learning rate) for the weight update.
        """
        super().__init__(num_taps=num_taps, mu=mu)
        self.reset()
        
    def reset(self) -> None:
        """Reset weights and delay line."""
        num_taps = self._params['num_taps']
        # Initialize weights (typically center tap is 1, rest 0)
        self.weights = np.zeros(num_taps, dtype=np.complex128)
        self.weights[num_taps // 2] = 1.0 + 0j 
        # Delay line for the FIR filter
        self.delay_line = np.zeros(num_taps, dtype=np.complex128)

    def _decision(self, symbol: complex) -> complex:
        """Simple decision device (e.g., QPSK hard decision)."""
        # Slices to the nearest QPSK constellation point: +/- 1 +/- 1j
        return np.sign(symbol.real) + 1j * np.sign(symbol.imag)

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        mu = self._params['mu']
        num_taps = self._params['num_taps']
        
        out_data = np.zeros_like(data)
        
        # Sample-by-sample feedback loop
        for i in range(len(data)):
            # 1. Shift new sample into the delay line
            self.delay_line = np.roll(self.delay_line, 1)
            self.delay_line[0] = data[i]
            
            # 2. Filter the signal (Inner product)
            # y = w^H * x
            y = np.vdot(self.weights, self.delay_line)
            out_data[i] = y
            
            # 3. Decision-Directed Error calculation
            d = self._decision(y)
            error = d - y
            
            # 4. Update Weights: w = w + mu * conj(error) * x
            self.weights += mu * np.conj(error) * self.delay_line
            
        return input_buf.clone_metadata(out_data)
```

## 2. Kalman Filter for Phase Tracking

While a PLL is a classical control-theory approach to phase tracking, the Kalman Filter is a statistical, optimal estimator. It explicitly models the Process Noise (e.g., oscillator linewidth) and Measurement Noise (e.g., AWGN).

### Mathematical Model (Scalar Phase Tracker)
1. **Predict State:** $\hat{x}_{n|n-1} = \hat{x}_{n-1|n-1}$ (Phase assumption: constant or linear drift)
2. **Predict Covariance:** $P_{n|n-1} = P_{n-1|n-1} + Q$ (where $Q$ is process noise)
3. **Kalman Gain:** $K_n = P_{n|n-1} / (P_{n|n-1} + R)$ (where $R$ is measurement noise)
4. **Update State:** $\hat{x}_{n|n} = \hat{x}_{n|n-1} + K_n (z_n - \hat{x}_{n|n-1})$ (where $z_n$ is the measured phase)
5. **Update Covariance:** $P_{n|n} = (1 - K_n) P_{n|n-1}$

### Python Implementation (`KalmanPhaseTracker`)

```python
class KalmanPhaseTracker(DSPBlock):
    """
    1D Kalman Filter to track and remove carrier phase noise.
    """
    def __init__(self, process_noise_q: float = 1e-4, measurement_noise_r: float = 1e-2) -> None:
        super().__init__(Q=process_noise_q, R=measurement_noise_r)
        self.reset()
        
    def reset(self) -> None:
        self.phase_est = 0.0  # State estimate (x)
        self.P = 1.0          # Error covariance estimate (P)

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        Q = self._params['Q']
        R = self._params['R']
        
        out_data = np.zeros_like(data)
        
        for i in range(len(data)):
            # --- 1. Predict Step ---
            # Assume phase is roughly the same as last step
            phase_pred = self.phase_est
            P_pred = self.P + Q
            
            # --- 2. Measurement ---
            # De-rotate the signal by predicted phase
            derotated = data[i] * np.exp(-1j * phase_pred)
            # Measure phase error (distance from nearest real axis / QPSK point)
            # simplified as the angle of the derotated symbol
            # (In a real system, you'd use a phase detector like Viterbi-Viterbi)
            measured_phase_error = np.angle(derotated)
            
            # The "measurement" z_n is the predicted phase + observed error
            z_n = phase_pred + measured_phase_error
            
            # --- 3. Update Step ---
            # Calculate Kalman Gain
            K = P_pred / (P_pred + R)
            
            # Update Phase Estimate
            self.phase_est = phase_pred + K * (z_n - phase_pred)
            
            # Keep phase bounded between -pi and pi
            self.phase_est = (self.phase_est + np.pi) % (2 * np.pi) - np.pi
            
            # Update Covariance
            self.P = (1 - K) * P_pred
            
            # Apply optimal phase correction to output
            out_data[i] = data[i] * np.exp(-1j * self.phase_est)
            
        return input_buf.clone_metadata(out_data)
```

## 3. C++ Migration Considerations

As established in the **Hybrid Performance Strategy**, adaptive algorithms are `for`-loop heavy and suffer greatly from Python's interpreter overhead. 

Once the mathematical behavior (convergence time, step sizes $\mu$, covariance bounds $Q/R$) is verified visually using the `DSPPlotter` utilities, these two blocks must be translated to C++. 

Because they maintain internal states (the `self.weights` array for LMS, and `self.phase_est` / `self.P` for Kalman), the C++ backend must be written as a **stateful C++ class** exposed via `pybind11`, rather than a simple stateless function.