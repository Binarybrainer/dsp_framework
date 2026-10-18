# 🔄 Phase-Locked Loop (PLL) Specification

The Phase-Locked Loop (PLL) is a fundamental feedback system used in digital communications to synchronize a local oscillator with the phase and frequency of an incoming carrier signal. 

Because the PLL relies heavily on a sample-by-sample feedback mechanism, it cannot be easily vectorized using standard NumPy array operations. Therefore, we design it as a stateful "Super-Block," making it a prime candidate for future migration to the C++ backend.

This document specifies the three core components of a digital PLL: the Phase Detector (PD), the Digital Loop Filter (LF), and the Numerically Controlled Oscillator (NCO).

## 1. Core Mathematical Model

### 1.1. Phase / Phase-Frequency Detector (PFD)
The phase detector computes the error between the incoming signal phase and the local NCO phase. For a complex baseband signal $x[n]$ and a local NCO output $y[n]$, a common and efficient phase error metric is the imaginary part of their conjugate product:

$$ \epsilon[n] = \text{Im}(x[n] \cdot y^*[n]) $$

Alternatively, a direct angle calculation can be used:

$$ \epsilon[n] = \arg(x[n]) - \arg(y[n]) $$

### 1.2. Digital Loop Filter (PI Controller)
The loop filter smoothens the phase error and drives the NCO. We use a Proportional-Integral (PI) controller to track both phase (proportional) and frequency (integral) offsets.

$$ v[n] = K_p \cdot \epsilon[n] + \text{integral\_state}[n] $$
$$ \text{integral\_state}[n+1] = \text{integral\_state}[n] + K_i \cdot \epsilon[n] $$

Where $K_p$ is the proportional gain and $K_i$ is the integral gain.

### 1.3. Numerically Controlled Oscillator (NCO)
The NCO generates the local complex exponential signal. Its phase is continuously updated by the loop filter's output.

$$ \theta[n+1] = \theta[n] + \omega_0 + v[n] $$
$$ y[n] = \exp(j \theta[n]) $$

Where $\omega_0$ is the free-running center frequency (often $0$ in baseband).

## 2. Python Implementation (`PLLBlock`)

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

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
```