# ⏱️ Clock and Data Recovery (CDR) & Eye Diagrams

In high-speed serial links (like PCIe, Ethernet, or SerDes), the receiver does not get a separate clock signal. It must extract the clock directly from the incoming data stream to know exactly when to sample the bits. This process is called **Clock and Data Recovery (CDR)**.

This document outlines the implementation of a digital CDR using an **Alexander (Bang-Bang) Phase Detector (BBPD)**, a **Phase Interpolator (PI)**, and an interactive **Eye Diagram** visualization tool to analyze signal integrity.

## 1. Alexander (Bang-Bang) Phase Detector (BBPD)

The BBPD is a non-linear phase detector that looks at three consecutive samples to determine if the sampling clock is Early, Late, or if there was no data transition.

* **$D_{n-1}$**: Previous Data sample
* **$E_n$**: Edge sample (taken exactly between $D_{n-1}$ and $D_n$)
* **$D_n$**: Current Data sample

**Logic Truth Table:**
* $D_{n-1} == D_n$: No transition. Output = `0` (Hold).
* $D_{n-1} \neq D_n$ AND $E_n == D_{n-1}$: Clock is **Early** (sampled before the transition). Output = `+1`.
* $D_{n-1} \neq D_n$ AND $E_n == D_n$: Clock is **Late** (sampled after the transition). Output = `-1`.

## 2. CDR "Super-Block" Implementation

To model the CDR in our DSP framework, we assume the input `SignalBuffer` is heavily oversampled (e.g., `sps = 16` or `32` samples per symbol). The **Phase Interpolator (PI)** acts as a pointer that shifts left or right across these sub-samples based on the BBPD and Loop Filter outputs, picking the absolute best sample to represent the data.

Because this requires sample-by-sample feedback, it is implemented as a Super-Block (and is a prime target for Phase 4 C++ acceleration).

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

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
```

## 3. Eye Diagram Visualization Utility

The Eye Diagram is the most critical tool for evaluating CDR performance and ISI (Intersymbol Interference). It works by slicing the continuous time-domain signal into chunks of 2 or 3 Unit Intervals (UIs) and overlaying them on top of each other.

We will add this to our existing `DSPPlotter` class from Phase 1.

```python
import plotly.graph_objects as go
import numpy as np
from signal_buffer_spec import SignalBuffer

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
```

## 4. Pipeline Integration Workflow

To utilize these tools, your testbench would look like this:

```python
# 1. Setup Channel with ISI/Crosstalk
channel = StaticMultipathChannel(taps=np.array([1.0, 0.3, -0.1]))
noise = AWGNChannel(snr_db=20)

# 2. Simulate Signal
# tx_signal is heavily oversampled (e.g., sps=16)
rx_signal = noise.process(channel.process(tx_signal))

# 3. Visualize the "Closed" Eye Diagram BEFORE the CDR
eye_fig = EyeDiagramPlotter.plot_eye(rx_signal, sps=16, num_uis=2)
eye_fig.write_html("pre_cdr_eye_diagram.html")

# 4. Recover Clock and Data
cdr = BangBangCDRBlock(sps=16, kp=0.05, ki=0.005)
recovered_symbols = cdr.process(rx_signal)

# 'recovered_symbols' is now running at 1 sample per symbol, 
# ready to be fed directly into the SymbolDemapperBlock!
```