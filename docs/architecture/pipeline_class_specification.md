# 🔗 DSPPipeline Class Specification

To build complex communication systems (like a full Tx-Channel-Rx chain), we need a way to connect individual `DSPBlock` instances sequentially. The `DSPPipeline` class achieves this by acting as a composite block. It takes an ordered list of blocks and routes the output of one block directly into the input of the next.

Since `DSPPipeline` itself inherits from `DSPBlock`, pipelines can be nested inside other pipelines, allowing for highly modular and hierarchical system designs.

## 1. Core Architecture

The `DSPPipeline` class must provide:
* **Sequential Execution**: Iterate through the internal list of blocks, updating the signal tensor at each step.
* **Global Reset Propagation**: When the pipeline is reset, it must propagate the `reset()` call down to every block it contains.
* **Dynamic Chaining**: Allow users to append new blocks to the pipeline after initialization.

## 2. Python Implementation

Here is the implementation of the `DSPPipeline` class.

```python
import numpy as np
from typing import List, Any
from dsp_block_api import DSPBlock

class DSPPipeline(DSPBlock):
    """
    A composite DSP block that chains multiple DSPBlocks together sequentially.
    """

    def __init__(self, blocks: List[DSPBlock] = None, **kwargs: Any) -> None:
        """
        Initialize the pipeline with an optional list of blocks.
        """
        super().__init__(**kwargs)
        self.blocks: List[DSPBlock] = blocks if blocks is not None else []

    def add_block(self, block: DSPBlock) -> None:
        """
        Append a new DSPBlock to the end of the pipeline.
        """
        if not isinstance(block, DSPBlock):
            raise TypeError("Only instances of DSPBlock can be added to the pipeline.")
        self.blocks.append(block)

    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Pass the input data through all blocks in the pipeline sequentially.
        
        Args:
            input_data (np.ndarray): The initial input signal.
            
        Returns:
            np.ndarray: The final processed signal.
        """
        current_data = input_data
        for block in self.blocks:
            # The output of block N becomes the input of block N+1
            current_data = block.process(current_data)
        
        return current_data

    def reset(self) -> None:
        """
        Reset the internal state of every block in the pipeline.
        """
        for block in getattr(self, 'blocks', []):
            block.reset()
            
    def update_params(self, **kwargs: Any) -> None:
        """
        Overrides the default update_params to broadcast parameter updates 
        to all child blocks. Note: This requires careful naming to avoid collisions.
        """
        super().update_params(**kwargs)
        for block in self.blocks:
            # Only update parameters that the specific block recognizes
            block_kwargs = {k: v for k, v in kwargs.items() if k in block.params}
            if block_kwargs:
                block.update_params(**block_kwargs)
```

## 3. Example Usage: Cascaded System

Here is an example demonstrating how to chain a sequence of basic blocks (assuming a `GainBlock` and a hypothetical `OffsetBlock` exist).

```python
# 1. Instantiate individual blocks
tx_gain = GainBlock(gain=2.0)
channel_offset = OffsetBlock(offset=0.5)
rx_gain = GainBlock(gain=0.5)

# 2. Create the pipeline
system = DSPPipeline([tx_gain, channel_offset, rx_gain])

# 3. Process a signal through the entire chain
input_signal = np.array([1.0, -1.0, 0.0])

# Math equivalent: y = ((x * 2.0) + 0.5) * 0.5
output_signal = system.process(input_signal)

print(f"Input:  {input_signal}")
print(f"Output: {output_signal}")

# 4. Reset all blocks simultaneously
system.reset()
```

## 4. Next Steps & Considerations

*   **Graph vs. Linear Pipeline**: `DSPPipeline` supports linear $A \rightarrow B \rightarrow C$ topologies. For complex topologies involving parallel paths or feedback loops (like a PLL where the output feeds back to a phase detector), we will either handle the loop explicitly inside a custom `PLLBlock`, or later design a `DSPGraph` class.