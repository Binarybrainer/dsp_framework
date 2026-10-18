# 📊 Interactive Visualization Utilities

In modern DSP framework development, static PNG images are often insufficient. When debugging a filter or analyzing a constellation diagram, you need to zoom in on specific samples, pan across frequency bands, and hover to see exact I/Q values.

This document outlines the solutions for interactive plotting and specifies a `Visualization` module that integrates directly with our `SignalBuffer`.

## 1. The Landscape of Python Plotting Solutions

To achieve interactive, savable plots, we evaluate the following ecosystem:

1. **Matplotlib:** 
   * *Best for:* Static academic papers (PDF/PNG). 
   * *Interactive?* Only within a Jupyter Notebook (via `%matplotlib widget`). Saving to an interactive file is difficult and clunky.
2. **Plotly (Recommended for Reports):** 
   * *Best for:* Interactive HTML reports. 
   * *Interactive?* Yes. It renders using WebGL/D3.js. You can call `fig.write_html("plot.html")` to generate a standalone file you can share with anyone.
3. **Bokeh / HoloViews:** 
   * *Best for:* Very large datasets. Like Plotly, exports to HTML.
4. **PyQtGraph:** 
   * *Best for:* Desktop GUI real-time streaming (as discussed in the GUI Spec). Cannot be easily saved as an interactive file to share.

**Decision:** We will build our plotting utilities using **Plotly** to satisfy the requirement for interactive, savable HTML files.

## 2. Core Visualization Class

We will create a `DSPPlotter` utility class. It takes a `SignalBuffer` and generates Plotly figure objects, which can either be displayed immediately (`fig.show()`) or saved (`fig.write_html()`).

### Dependencies
```bash
pip install plotly scipy
```

### 3. Time-Domain Utility

Visualizes the real (In-phase) and imaginary (Quadrature) components of a signal over time. It leverages the `time_vector()` method we built into `SignalBuffer`.

```python
import plotly.graph_objects as go
import numpy as np
from signal_buffer_spec import SignalBuffer

class DSPPlotter:
    
    @staticmethod
    def plot_time_domain(buffer: SignalBuffer, title: str = "Time Domain Signal") -> go.Figure:
        """Generates an interactive Time-Domain plot."""
        t = buffer.time_vector()
        
        fig = go.Figure()
        
        # Plot Real (I) component
        fig.add_trace(go.Scattergl(
            x=t, y=buffer.data.real, 
            mode='lines', name='Real (I)',
            line=dict(color='blue')
        ))
        
        # Plot Imaginary (Q) component if signal is complex
        if np.iscomplexobj(buffer.data):
            fig.add_trace(go.Scattergl(
                x=t, y=buffer.data.imag, 
                mode='lines', name='Imaginary (Q)',
                line=dict(color='red')
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            hovermode="x unified" # Shows both I and Q values when hovering
        )
        return fig
```

### 4. Frequency-Domain (PSD) Utility

For frequency analysis, we should not just plot the raw FFT, as it is extremely noisy. Standard DSP practice is to compute the **Power Spectral Density (PSD)** using Welch's method to average out the noise floor.

```python
from scipy.signal import welch

    @staticmethod
    def plot_psd(buffer: SignalBuffer, nperseg: int = 1024, title: str = "Power Spectral Density") -> go.Figure:
        """Generates an interactive PSD plot using Welch's method."""
        fs = buffer.sample_rate
        
        # Calculate PSD using SciPy
        # return_onesided=False ensures we see negative frequencies for complex baseband signals
        frequencies, psd_linear = welch(
            buffer.data, 
            fs=fs, 
            nperseg=nperseg, 
            return_onesided=not np.iscomplexobj(buffer.data)
        )
        
        # Shift zero frequency to the center
        frequencies = np.fft.fftshift(frequencies)
        psd_linear = np.fft.fftshift(psd_linear)
        
        # Convert to dB/Hz
        psd_db = 10 * np.log10(psd_linear + 1e-12) # Add epsilon to prevent log(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=frequencies, y=psd_db, 
            mode='lines', name='PSD',
            line=dict(color='darkgreen')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power Spectral Density (dB/Hz)",
            yaxis=dict(autorange="reversed") if np.max(psd_db) < 0 else {}
        )
        return fig
```

### 5. Constellation Diagram Utility

A constellation diagram is a scatter plot of the complex data ($I$ vs $Q$). 
*Optimization Note:* Web browsers crash if you try to render millions of scatter points. We use `Scattergl` (WebGL) and limit the maximum number of plotted points to maintain interactive performance.

```python
    @staticmethod
    def plot_constellation(buffer: SignalBuffer, max_points: int = 50000, title: str = "Constellation Diagram") -> go.Figure:
        """Generates an interactive I/Q Constellation Diagram."""
        
        # Decimate data if it exceeds max_points to prevent browser freeze
        data_to_plot = buffer.data
        if len(data_to_plot) > max_points:
            step = len(data_to_plot) // max_points
            data_to_plot = data_to_plot[::step]
            title += f" (Decimated to {max_points} pts)"
            
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=data_to_plot.real, 
            y=data_to_plot.imag, 
            mode='markers', 
            name='Symbols',
            marker=dict(
                color='rgba(0, 50, 255, 0.4)', # Semi-transparent blue
                size=3,
                line=dict(width=0)
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="In-Phase (I)",
            yaxis_title="Quadrature (Q)",
            width=700, height=700, # Keep aspect ratio square
            xaxis=dict(scaleanchor="y", scaleratio=1), # Force square axes physically
        )
        return fig
```

## 6. How to Use and Save as HTML

Once you process a signal through your pipeline, you can generate and save these interactive plots effortlessly:

```python
# Assuming 'rx_signal' is a SignalBuffer output from your pipeline
plotter = DSPPlotter()

# 1. Create the figures
fig_time = plotter.plot_time_domain(rx_signal)
fig_psd = plotter.plot_psd(rx_signal)
fig_const = plotter.plot_constellation(rx_signal)

# 2. Display them instantly in Jupyter Notebook or Browser
# fig_const.show()

# 3. SAVE as standalone interactive HTML files
fig_time.write_html("output_time_domain.html")
fig_psd.write_html("output_spectrum.html")
fig_const.write_html("output_constellation.html")
```