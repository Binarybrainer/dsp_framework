import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from dsp_framework.core import SignalBuffer
from scipy.signal import windows

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
    def plot_fft(buffer: SignalBuffer, nperseg: int = 1024, title: str = "Fast Fourier Transform", figsize: tuple = (10,4), window_type = "hann", normalization = "amplitude") -> plt.Figure:

        """
        window_type : rectangular, hann, hamming, blackman, flattop
        normalization: amplitude, ortho
        
        """
        def _get_window(length: int) -> np.ndarray:
            w_type = window_type
            if w_type == 'rectangular' or w_type == 'none':
                return np.ones(length, dtype=np.float64)
            elif w_type == 'hann':
                return windows.hann(length, sym=False)
            elif w_type == 'hamming':
                return windows.hamming(length, sym=False)
            elif w_type == 'blackman':
                return windows.blackman(length, sym=False)
            elif w_type == 'flattop':
                return windows.flattop(length, sym=False)
            else:
                raise ValueError(f"Unsupported window type: {w_type}")

        data = buffer.data
        N = len(data)
        n_fft = nperseg 
        
        # 1. Apply Window Function
        w = _get_window(N)
        windowed_data = data * w
        
        # 2. Perform Forward FFT
        norm_mode = normalization
        if norm_mode == 'amplitude':
            # Coherent amplitude scaling
            scale = 1.0 / np.sum(w)
            fft_data = np.fft.fft(windowed_data, n=n_fft) * scale
        elif norm_mode == 'ortho':
            fft_data = np.fft.fft(windowed_data, n=n_fft, norm='ortho')
        else:
            fft_data = np.fft.fft(windowed_data, n=n_fft)

        
        # Frequency step
        df = buffer.sample_rate / n_fft
        # Frequency
        frequencies = np.arange(0,buffer.sample_rate, df)

        # Shift zero frequency to center
        fft_data = np.fft.fftshift(fft_data)

        # Convert to dB scale (with epsilon guard against log(0))
        fft_data_db = 10 * np.log10(fft_data + 1e-12)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(frequencies, fft_data_db, color='#2ca02c', linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Amplitude (dB/Hz)", fontsize=10)
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
        
        ax.scatter(data.real, data.imag, color="#d62727", alpha=0.3, s=12, edgecolors='none')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("In-Phase (I)", fontsize=10)
        ax.set_ylabel("Quadrature (Q)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Enforce equal aspect ratio to prevent geometric distortion of constellation rings
        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()
        return fig