# 🔄 Feedback Loops in Linear Pipelines (PLL & CDR)

When using a linear `DSPPipeline` (where data flows strictly from Block A $\rightarrow$ Block B), implementing feedback loops like Phase-Locked Loops (PLLs) or Clock and Data Recovery (CDR) circuits presents a unique challenge. In a feedback loop, the processing of sample $n$ depends directly on the internal state calculated during sample $n-1$.

To solve this, we use the **"Super-Block" Design Pattern**.

## 1. The "Super-Block" Design Pattern

Instead of trying to wire the output of one `DSPBlock` back to the input of a previous one via the pipeline, we encapsulate the *entire feedback loop* inside a single, composite `DSPBlock` (the "Super-Block").

Inside this block's `process()` method, we abandon block-level vectorization and instead implement a **sample-by-sample `for` loop**. 

### Anatomy of a PLL Super-Block
A standard PLL consists of three sub-components:
1. **Phase Detector (PD)**
2. **Loop Filter (LF)** (Usually a Proportional-Integral controller)
3. **Numerically Controlled Oscillator (NCO)**

Instead of making these three separate blocks in a pipeline, we write one `PLLBlock`. Inside its loop, it computes the PD, updates the LF, and steps the NCO sequentially for every single sample.

## 2. Python Implementation: `PLLBlock`

Here is an example of a carrier recovery PLL designed to track the phase of a complex baseband signal. It inherits perfectly from our `DSPBlock` ABC, meaning it can be dropped right into a `DSPPipeline`.

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock

class PLLBlock(DSPBlock):
    """
    A Phase-Locked Loop (PLL) 'Super-Block' for carrier phase recovery.
    It encapsulates the Phase Detector, Loop Filter, and NCO internally.
    """
    
    def __init__(self, kp: float = 0.1, ki: float = 0.01, nominal_freq: float = 0.0) -> None:
        super().__init__(kp=kp, ki=ki, nominal_freq=nominal_freq)
        self.reset()
        
    def reset(self) -> None:
        """Reset the internal states of the PLL."""
        self.phase_acc = 0.0          # NCO phase accumulator
        self.loop_filter_int = 0.0    # Loop filter integrator state
        
    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Process the input array sample-by-sample to maintain the feedback loop.
        """
        # Pre-allocate output array
        output_data = np.zeros_like(input_data)
        
        # Extract parameters for fast lookup in the loop
        kp = self._params['kp']
        ki = self._params['ki']
        f_nom = self._params['nominal_freq']
        
        # Sample-by-sample feedback loop
        for i in range(len(input_data)):
            # 1. NCO: Generate current local oscillator sample
            nco_out = np.exp(1j * self.phase_acc)
            
            # 2. Phase Detector: Calculate phase error (e.g., using angle difference)
            # We conjugate the NCO output to "de-rotate" the input
            error = np.angle(input_data[i] * np.conj(nco_out))
            
            # 3. Loop Filter: Proportional-Integral (PI) controller
            self.loop_filter_int += ki * error
            control_signal = (kp * error) + self.loop_filter_int
            
            # 4. Update NCO Phase for the *next* sample (n+1)
            self.phase_acc += f_nom + control_signal
            
            # Keep phase bounded between -pi and pi (optional but good practice)
            self.phase_acc = (self.phase_acc + np.pi) % (2 * np.pi) - np.pi
            
            # Assign de-rotated output
            output_data[i] = input_data[i] * np.conj(nco_out)
            
        return output_data
```

## 3. Why this Architecture is Perfect for our Roadmap

Writing `for` loops in pure Python (NumPy) over large arrays is notoriously slow. However, this is exactly what we want in our **Phase 4: Profiling & Backend Migration** step!

1. **Phase 1-3:** We prototype this exact `PLLBlock` in Python to verify the math ($K_p$, $K_i$ tuning, loop stability).
2. **Phase 4:** Because the entire feedback loop is cleanly isolated inside the `PLLBlock.process()` method, it becomes the perfect candidate to rewrite in C++ using `pybind11`. 
3. **Result:** We replace the slow Python `for` loop with a blazing-fast C++ `for` loop, while the Python user still just calls `my_pll.process(data)` as normal.