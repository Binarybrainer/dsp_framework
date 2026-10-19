# ⏱️ Performance Profiling Specification

Before migrating any Python code to C++, it is critical to quantitatively identify the actual bottlenecks in the `DSPPipeline`. Optimizing the wrong block wastes engineering time and adds unnecessary complexity.

This document specifies the integration of `cProfile` for macro-level pipeline analysis and `line_profiler` for micro-level function analysis.

## 1. Macro-Profiling with `cProfile`

`cProfile` is a built-in Python module that tracks the total execution time spent in every function. We use it to identify the top 3 slowest blocks in the entire DSP chain.

### 1.1. Profiling Script Setup

Create a dedicated profiling script (e.g., `profile_pipeline.py`) that runs a heavy simulation (e.g., 1 million samples) through the full pipeline.

```python
import cProfile
import pstats
import io
import numpy as np

# Import your framework components
from signal_buffer_spec import SignalBuffer
from dsp_block_api import DSPPipeline
from pll_specification import PLLBlock
from modulation_demodulation_spec import SymbolMapperBlock
from channel_modeling_spec import AWGNChannel

def run_simulation_benchmark():
    # 1. Generate large dataset
    N_samples = 1000000
    tx_symbols = np.random.randint(0, 4, N_samples)
    tx_buf = SignalBuffer(tx_symbols, sample_rate=1e6)
    
    # 2. Build Pipeline
    mapper = SymbolMapperBlock(modulation_type='QPSK')
    channel = AWGNChannel(snr_db=15.0)
    pll = PLLBlock(kp=0.05, ki=0.005)
    
    pipeline = DSPPipeline([mapper, channel, pll])
    
    # 3. Execute
    rx_buf = pipeline.process(tx_buf)

if __name__ == "__main__":
    # Setup Profiler
    pr = cProfile.Profile()
    pr.enable()
    
    run_simulation_benchmark()
    
    pr.disable()
    
    # Sort and print results
    s = io.StringIO()
    sortby = pstats.SortKey.CUMTIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    
    print("=== TOP 10 SLOWEST FUNCTIONS ===")
    ps.print_stats(10) # Print top 10 bottlenecks
    print(s.getvalue())
```

### 1.2. Identifying the Culprits

When running the script, look for the `cumtime` (cumulative time) column. 
You will typically find that **sample-by-sample feedback loops** dominate the execution time:
1. `PLLBlock.process` (Heavy for-loop, trigonometric functions).
2. `BangBangCDRBlock.process` (Stateful phase interpolation).
3. `LMS_EqualizerBlock.process` (Weight updates per sample).

Conversely, vectorized blocks like `AWGNChannel` or `SymbolMapperBlock` will barely register on the profile.

## 2. Micro-Profiling with `line_profiler`

Once `cProfile` identifies a slow block (e.g., `PLLBlock`), we use `line_profiler` to see exactly which line of Python code inside that block's `process()` method is causing the delay.

### 2.1. Installation

```bash
pip install line_profiler
```

### 2.2. Decorating the Target Function

Add the `@profile` decorator to the specific method you want to analyze. (Note: You do not need to import `profile`, the module injects it at runtime).

```python
# Inside pll_specification.py

class PLLBlock(DSPBlock):
    # ... init and reset ...

    @profile  # <--- Add this decorator
    def process(self, input_buf: SignalBuffer, **kwargs) -> SignalBuffer:
        data = input_buf.data
        kp = self._params['kp']
        # ... rest of the loop ...
```

### 2.3. Executing the Line Profiler

Run the simulation script using the `kernprof` tool provided by `line_profiler`:

```bash
kernprof -l -v profile_pipeline.py
```

### 2.4. Analyzing the Output

The output will display the target function line-by-line, showing the `% Time` spent on each line. 

*Example Output:*
```text
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    38                                           @profile
    39                                           def process(self, input_buf):
    ...
    46   1000000   150000.0      0.1     15.0      nco_out = np.exp(1j * self.phase_state)
    47   1000000   300000.0      0.3     30.0      phase_error = np.imag(data[i] * np.conj(nco_out))
    ...
```

## 3. Next Steps: The C++ Migration

Once you have identified the exact lines causing the bottleneck (usually NumPy scalar operations inside a massive Python `for` loop), those specific blocks are flagged for Phase 4: rewriting the loop in C++ and binding it back via `pybind11`.