# 🌟 Co-Packaged Optics (CPO) Simulation Specification

As data rates scale to 1.6T and 3.2T per switch port, routing high-speed electrical signals over copper PCBs becomes infeasible due to extreme insertion loss and crosstalk. **Co-Packaged Optics (CPO)** solves this by moving the electro-optic transceivers directly onto the same organic substrate as the host ASIC.

To simulate CPO in our DSP framework, we must model the transition from the electrical domain to the optical domain and back. This involves simulating Continuous Wave (CW) lasers, optical modulators, short-reach fiber physics, and photodetector receivers.

## 1. System Architecture and Mathematical Models

For short-reach CPO (e.g., intra-datacenter links up to 500m), **Intensity Modulation with Direct Detection (IM/DD)** using PAM4 is the industry standard.

### 1.1. Electro-Optic Modulation (Transmitter)

An external CW laser provides a constant optical carrier. The electrical PAM4 signal drives a **Mach-Zehnder Modulator (MZM)** to vary the optical intensity.

*   **Laser Power & RIN:** The laser has an average optical power $P_0$ and suffers from Relative Intensity Noise (RIN).
*   **MZM Transfer Function:** The output optical power depends non-linearly on the drive voltage $V(t)$ relative to the modulator's switching voltage ($V_{\pi}$):
    $$ P_{out}(t) = P_0 \cdot \alpha_{IL} \cdot \cos^2\left(\frac{\pi}{2} \frac{V(t) - V_{bias}}{V_{\pi}}\right) $$
    *(Where $\alpha_{IL}$ is the insertion loss).*

### 1.2. Optical Channel

For CPO, the fiber is typically single-mode (SMF) or multi-mode (MMF).
*   **Attenuation:** Signal power drops exponentially with distance.
*   **Chromatic Dispersion (CD):** Different wavelengths travel at different speeds, causing pulse broadening. For very short CPO links (< 100m), this is often negligible, but included for completeness.

### 1.3. Opto-Electrical Conversion (Receiver)

A Photodiode (PD) converts the received optical power back into an electrical current, amplified by a Transimpedance Amplifier (TIA).
*   **Responsivity ($\mathcal{R}$):** Conversion efficiency (Amps/Watt).
*   **Receiver Noise:** Dominated by **Thermal Noise** (from the TIA) and **Shot Noise** (quantum nature of light).

## 2. Python Implementation (`CoPackagedOpticsBlock`)

We implement the entire IM/DD link as a single `DSPBlock` that takes an electrical `SignalBuffer` (voltage) and outputs an electrical `SignalBuffer` (voltage), encapsulating the optical physics internally.

```python
import numpy as np
from typing import Any
from dsp_framework.core.dsp_block_api import DSPBlock
from dsp_framework.core.signal_buffer_spec import SignalBuffer

class CoPackagedOpticsBlock(DSPBlock):
    """
    Simulates a complete IM/DD Co-Packaged Optics (CPO) link including 
    MZM modulation, fiber attenuation, and Photodiode/TIA reception.
    """
    def __init__(self, 
                 laser_power_dbm: float = 10.0,
                 v_pi: float = 2.0,
                 extinction_ratio_db: float = 6.0,
                 fiber_length_km: float = 0.1,
                 fiber_attenuation_db_km: float = 0.3,
                 pd_responsivity: float = 0.8,
                 tia_noise_density: float = 1e-11) -> None:
        """
        Args:
            laser_power_dbm: CW Laser output power.
            v_pi: Voltage required to swing the MZM from max to min transmission.
            extinction_ratio_db: Target ER to calculate V_bias.
            fiber_length_km: Length of the optical fiber.
            pd_responsivity: Photodiode responsivity in A/W.
            tia_noise_density: Thermal noise current density (A/sqrt(Hz)).
        """
        super().__init__(
            laser_power_dbm=laser_power_dbm,
            v_pi=v_pi,
            extinction_ratio_db=extinction_ratio_db,
            fiber_length_km=fiber_length_km,
            fiber_attenuation_db_km=fiber_attenuation_db_km,
            pd_responsivity=pd_responsivity,
            tia_noise_density=tia_noise_density
        )
        self.reset()
        
    def reset(self) -> None:
        pass

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        # 1. Electrical Drive Signal (V)
        V_drive = np.real(input_buf.data) # IM/DD uses real baseband signals
        fs = input_buf.sample_rate
        
        # Extract parameters
        P0 = 10 ** (self._params['laser_power_dbm'] / 10.0) * 1e-3 # Convert dBm to Watts
        V_pi = self._params['v_pi']
        R_pd = self._params['pd_responsivity']
        
        # 2. Electro-Optic Modulation (MZM)
        # Assume ideal bias for PAM4 to utilize the linear region of the cosine curve
        V_bias = V_pi / 2.0
        
        # MZM Non-linear transfer function
        # P_opt = P0 * cos^2(pi/2 * (V + V_bias) / V_pi)
        phase_shift = (np.pi / 2.0) * ((V_drive + V_bias) / V_pi)
        P_opt_tx = P0 * (np.cos(phase_shift) ** 2)
        
        # 3. Optical Channel (Fiber)
        # Attenuation
        alpha_db = self._params['fiber_length_km'] * self._params['fiber_attenuation_db_km']
        loss_linear = 10 ** (-alpha_db / 10.0)
        P_opt_rx = P_opt_tx * loss_linear
        
        # (Optional: Add Chromatic Dispersion using FFT here if simulating >2km links)
        
        # 4. Opto-Electrical Receiver (PD + TIA)
        # Convert power to current
        I_rx = P_opt_rx * R_pd
        
        # Add Thermal Noise (Gaussian)
        # Bandwidth is roughly fs/2. Noise variance = PSD^2 * BW
        bw = fs / 2.0
        sigma_thermal = self._params['tia_noise_density'] * np.sqrt(bw)
        thermal_noise = np.random.normal(0, sigma_thermal, len(I_rx))
        
        # Add Shot Noise (Signal dependent)
        q = 1.602e-19 # Electron charge
        sigma_shot = np.sqrt(2 * q * I_rx * bw)
        shot_noise = np.random.normal(0, sigma_shot)
        
        # Total recovered electrical signal (assuming 1 Ohm TIA transimpedance for normalized voltage)
        V_rx = I_rx + thermal_noise + shot_noise
        
        return input_buf.clone_metadata(V_rx)
```

## 3. Integration in the DSP Pipeline

This block can be sandwiched right between your TX DSP and RX DSP components. It naturally imparts the non-linear compression (from the MZM's cosine curve) and signal-dependent noise that real neural equalizers (like the one defined in Phase 4) are designed to combat.

```python
# Create the CPO Optical Link
cpo_link = CoPackagedOpticsBlock(laser_power_dbm=13.0, fiber_length_km=0.05)

# Simulate a 224G PAM4 transmission over the optical link
# tx_electrical is the output of the DAC
rx_electrical = cpo_link.process(tx_electrical)

# The rx_electrical is then fed into the TimeInterleavedADCBlock 
# and subsequently the NeuralEqualizerBlock!
```