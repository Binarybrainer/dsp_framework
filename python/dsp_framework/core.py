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


class DSPPipeline(DSPBlock):
    """
    A composite DSP block that chains multiple DSPBlocks together sequentially.
    """

    def __init__(self, blocks: list[DSPBlock] = None, **kwargs: Any) -> None:
        """
        Initialize the pipeline with an optional list of blocks.
        """
        super().__init__(**kwargs)
        self.blocks: list[DSPBlock] = blocks if blocks is not None else []

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

class SignalBuffer:
    """
    A smart wrapper around NumPy arrays to carry DSP metadata (sample rate, time)
    through a processing pipeline.
    """
    
    def __init__(self, 
                 data: np.ndarray, 
                 sample_rate: float = 1.0, 
                 timestamp: float = 0.0):
        """
        Initialize the SignalBuffer.
        
        Args:
            data (np.ndarray): The 1D or 2D array of signal samples.
            sample_rate (float): The sampling frequency in Hz.
            timestamp (float): The absolute time of the first sample.
        """
        self.data: np.ndarray = np.asarray(data)
        self.sample_rate: float = sample_rate
        self.timestamp: float = timestamp

    @property
    def active_length(self) -> int:
        """Return the number of samples in the primary axis."""
        return self.data.shape[0]

    def time_vector(self) -> np.ndarray:
        """
        Generate a time vector matching the current data length.
        Useful for plotting or time-dependent channel models.
        """
        dt = 1.0 / self.sample_rate
        return self.timestamp + np.arange(self.active_length) * dt

    def clone_metadata(self, new_data: Optional[np.ndarray] = None) -> 'SignalBuffer':
        """
        Create a new SignalBuffer with the same metadata but different data.
        If new_data is not provided, it creates an empty buffer of the same shape.
        """
        if new_data is None:
            new_data = np.zeros_like(self.data)
            
        return SignalBuffer(
            data=new_data,
            sample_rate=self.sample_rate,
            timestamp=self.timestamp
        )

    def advance_time(self) -> None:
        """
        Advance the timestamp by the length of the current data buffer.
        Used when operating in a continuous streaming loop.
        """
        duration = self.active_length / self.sample_rate
        self.timestamp += duration

    def __repr__(self) -> str:
        return (f"SignalBuffer(length={self.active_length}, "
                f"fs={self.sample_rate}Hz, t0={self.timestamp}s, dtype={self.data.dtype})")


from typing import Any
import numpy as np

class SignalSource(DSPBlock):
    """Base class for all signal generators."""
    def __init__(self, sample_rate_hz: float, **kwargs: Any) -> None:
        super().__init__(sample_rate_hz=sample_rate_hz, **kwargs)
        self.reset()
        
    def reset(self) -> None:
        """Reset the internal timestamp for continuous generation."""
        self.current_time = 0.0

    def generate(self, num_samples: int) -> SignalBuffer:
        """
        Generate a SignalBuffer of the requested length.
        Must be implemented by child classes.
        """
        pass

from scipy.signal import max_len_seq

class PRBSGenerator(SignalSource):
    """
    Generates a Pseudo-Random Binary Sequence (PRBS).
    Outputs a SignalBuffer containing binary data (0s and 1s).
    """
    def __init__(self, sample_rate_hz: float, prbs_order: int = 7) -> None:
        """
        prbs_order: e.g., 7 for PRBS7 (period = 2^7 - 1), 15 for PRBS15.
        """
        super().__init__(sample_rate_hz=sample_rate_hz, prbs_order=prbs_order)
        self._generate_full_sequence()
        
    def _generate_full_sequence(self) -> None:
        """Pre-generate the entire PRBS period to loop through."""
        order = self._params['prbs_order']
        # scipy max_len_seq returns the sequence and the state. We only need the sequence.
        self.full_seq, _ = max_len_seq(order)
        self.seq_length = len(self.full_seq)
        self.seq_index = 0

    def update_params(self, **kwargs: Any) -> None:
        super().update_params(**kwargs)
        if 'prbs_order' in kwargs:
            self._generate_full_sequence()

    def generate(self, num_samples: int) -> SignalBuffer:
        # Repeat the pre-generated sequence to satisfy the requested length
        repeats = (num_samples // self.seq_length) + 2
        tiled_seq = np.tile(self.full_seq, repeats)
        
        # Extract the exact chunk we need
        out_data = tiled_seq[self.seq_index : self.seq_index + num_samples]
        
        # Update index for the next chunk
        self.seq_index = (self.seq_index + num_samples) % self.seq_length
        
        # Create SignalBuffer
        buf = SignalBuffer(
            data=out_data.astype(np.float32), 
            sample_rate=self._params['sample_rate_hz'],
            timestamp=self.current_time
        )
        
        # Advance internal time
        self.current_time += num_samples / self._params['sample_rate_hz']
        return buf