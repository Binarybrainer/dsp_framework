# 🧩 DSPBlock Abstract Base Class Specification

In our Python-First DSP framework, the `DSPBlock` serves as the foundational Abstract Base Class (ABC) for all signal processing components. By enforcing a standard interface, we ensure that blocks (like Modulators, Filters, and PLLs) can be easily chained together in a pipeline and eventually swapped with C++ back-ends without breaking the system architecture.

## 1. Core Philosophy

Every DSP block must adhere to a predictable lifecycle and processing pattern:
*   **Initialization**: Setup default parameters and allocate any necessary internal state variables.
*   **Processing (`process`)**: Accept input arrays, apply the DSP algorithm, update internal states, and return the output arrays.
*   **State Management (`reset`)**: Clear any historical states (e.g., filter delay lines, PLL phase accumulators) for a fresh simulation run.
*   **Dynamic Configuration (`update_params`)**: Allow on-the-fly tuning of block parameters without destroying and recreating the object.

## 2. Python Implementation

Below is the foundational Python code using the built-in `abc` module.

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

class DSPBlock(ABC):
    """
    Abstract Base Class for all Digital Signal Processing blocks in the framework.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the DSP block with arbitrary parameters.
        Child classes should call super().__init__(**kwargs) to register parameters.
        """
        self._params: Dict[str, Any] = kwargs
        self.reset()

    @abstractmethod
    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Core signal processing routine.
        
        Args:
            input_data (np.ndarray): The input signal array.
            **kwargs: Optional block-specific processing flags.
            
        Returns:
            np.ndarray: The processed output signal array.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state/memory of the DSP block.
        This must be implemented by child classes to clear delay lines, 
        accumulators, or state machine variables.
        """
        pass

    def update_params(self, **kwargs: Any) -> None:
        """
        Update the configuration parameters of the block on the fly.
        
        Args:
            **kwargs: Key-value pairs of parameters to update.
            
        Raises:
            KeyError: If a parameter being updated does not exist in the block.
        """
        for key, value in kwargs.items():
            if key in self._params:
                self._params[key] = value
                # Child classes can override this method to trigger specific 
                # recalculations (e.g., recalculating filter coefficients) 
                # when a parameter changes.
            else:
                raise KeyError(f"Parameter '{key}' is not a valid parameter for {self.__class__.__name__}.")

    @property
    def params(self) -> Dict[str, Any]:
        """Get the current parameters of the block."""
        return self._params
```

## 3. Example Usage: A Simple Gain Block

Here is how a developer would use `DSPBlock` to create a concrete implementation.

```python
class GainBlock(DSPBlock):
    """A simple DSP block that applies a scalar gain to the input signal."""
    
    def __init__(self, gain: float = 1.0) -> None:
        # Register the gain parameter
        super().__init__(gain=gain)
        
    def process(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        # Multiply input by the gain parameter
        return input_data * self._params['gain']
        
    def reset(self) -> None:
        # Gain block has no historical state (memoryless), so pass
        pass
        
    def update_params(self, **kwargs: Any) -> None:
        # Utilize the parent implementation to update the dictionary
        super().update_params(**kwargs)
        print(f"Gain updated to: {self._params['gain']}")

# Usage:
# my_gain = GainBlock(gain=2.0)
# out = my_gain.process(np.array([1.0, 2.0]))
# my_gain.update_params(gain=5.0)
```