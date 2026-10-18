# 📊 Matplotlib Visualization Utilities Specification

While Plotly handles interactive HTML exports for exploration, technical reports, academic papers, and fast debugging scripts often require traditional, high-resolution static plotting tools. 

This document defines the `MatplotlibDSPPlotter` utility class, which natively consumes our `SignalBuffer` objects to render time-domain waveforms, frequency-domain spectra (PSD), and I/Q constellation scatter plots.

## 1. Core Architecture

The `MatplotlibDSPPlotter` wraps standard `matplotlib.pyplot` figures. It extracts data, timestamps, and sample rates directly from the `SignalBuffer` wrapper without requiring manual parameter calculations.

### Dependencies

```bash
pip install matplotlib numpy scipy
```

## 2. Python Implementation (`MatplotlibDSPPlotter`)

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from signal_buffer_spec import SignalBuffer

class MatplotlibDSPPlotter:
    """
    Utility class for rendering static, publication-quality DSP plots 
    using Matplotlib, directly integrated with SignalBuffer.
    """

    @staticmethod
    def plot_time_domain(buffer: SignalBuffer, title: str = "Time Domain Signal", figsize: tuple = (10, 4)) -> plt.Figure:
        """
        Renders the real (In-phase) and imaginary (Quadrature) parts of a signal over time.
        """
        t = buffer.time_vector()
        data = buffer.data

        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Real (I) component
        ax.plot(t, data.real, label='Real (I)', color='#1f77b4', alpha=0.9, linewidth=1.2)
        
        # Plot Imaginary (Q) component if complex
        if np.iscomplexobj(data):
            ax.plot(t, data.imag, label='Imaginary (Q)', color='#ff7f0e', alpha=0.8, linewidth=1.2, linestyle='--')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_psd(buffer: SignalBuffer, nperseg: int = 1024, title: str = "Power Spectral Density", figsize: tuple = (10, 4)) -> plt.Figure:
        """
        Computes and renders the Power Spectral Density (PSD) using Welch's method.
        """
        fs = buffer.sample_rate
        data = buffer.data

        # Compute Welch PSD (supports complex baseband with two-sided spectrum)
        frequencies, psd_linear = welch(
            data,
            fs=fs,
            nperseg=nperseg,
            return_onesided=not np.iscomplexobj(data)
        )

        # Shift zero frequency to center
        frequencies = np.fft.fftshift(frequencies)
        psd_linear = np.fft.fftshift(psd_linear)

        # Convert to dB scale (with epsilon guard against log(0))
        psd_db = 10 * np.log10(psd_linear + 1e-12)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frequencies, psd_db, color='#2ca02c', linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("PSD (dB/Hz)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_constellation(buffer: SignalBuffer, max_points: int = 10000, title: str = "Constellation Diagram", figsize: tuple = (6, 6)) -> plt.Figure:
        """
        Renders an I/Q scatter plot for digital communication constellations.
        Includes automatic downsampling for massive symbol blocks.
        """
        data = buffer.data
        if not np.iscomplexobj(data):
            raise ValueError("Constellation plotting requires a complex-valued SignalBuffer.")

        # Decimate if data exceeds max points to protect renderer memory
        if len(data) > max_points:
            step = len(data) // max_points
            data = data[::step]
            title += f" (Decimated)"

        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(data.real, data.imag, color='#d62728', alpha=0.3, s=12, edgecolors='none')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("In-Phase (I)", fontsize=10)
        ax.set_ylabel("Quadrature (Q)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Enforce equal aspect ratio to prevent geometric distortion of constellation rings
        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()
        return fig
```

## 3. Usage Example in Scripts

```python
# Assuming 'rx_signal' is a SignalBuffer output from your pipeline
plotter = MatplotlibDSPPlotter()

# 1. Generate Figures
fig_time = plotter.plot_time_domain(rx_signal, title="Received Waveform Time Domain")
fig_psd = plotter.plot_psd(rx_signal, title="Channel Output Spectrum")
fig_const = plotter.plot_constellation(rx_signal, title="Demodulated 16-QAM Constellation")

# 2. Save to high-res formats for documents or presentations
fig_time.savefig("time_domain_report.png", dpi=300)
fig_psd.savefig("psd_report.png", dpi=300)
fig_const.savefig("constellation_report.pdf", format="pdf")

# Or display interactively using GUI backends
plt.show()
```