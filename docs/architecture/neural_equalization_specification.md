# 🧠 Deep Learning Equalization (Neural DSP)

As baud rates push towards 200G/800G per lane in electrical SerDes, and optical networks face severe Kerr non-linearities, traditional equalizers like the Decision Feedback Equalizer (DFE) or Volterra series struggle. They are either unable to model complex non-linear cross-talk or become mathematically intractable.

This document specifies the integration of **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** to replace classical equalizers. By leveraging PyTorch, we can deploy Neural Equalizers as standard `DSPBlock` instances in our pipeline.

## 1. Architectural Strategy: PyTorch Integration

Unlike the LMS adaptive filter which updates its weights sample-by-sample in real-time, Neural Equalizers typically operate in two distinct phases:

1. **Offline Training Phase:** A model is trained using a massive dataset of known transmitted symbols (labels) and heavily distorted received waveforms (features).
2. **Inference Phase:** The pre-trained weights are loaded into our DSP pipeline to process streaming data chunk-by-chunk.

Because our `SignalBuffer` relies on NumPy arrays, transitioning to PyTorch Tensors for the inference phase is practically a zero-copy operation, maintaining high throughput.

## 2. Model Design: 1D CNN Equalizer

A 1D CNN is highly effective for equalization because it behaves like a non-linear FIR filter. It slides a receptive field across the time-domain signal, capturing both Linear Inter-Symbol Interference (ISI) and non-linear distortion.

### PyTorch CNN Architecture

```python
import torch
import torch.nn as nn

class CNN1DEqualizerModel(nn.Module):
    """
    A 1D Convolutional Neural Network designed to equalize non-linear PAM4/PAM8 signals.
    Acts as a non-linear substitute for a Feed-Forward Equalizer (FFE).
    """
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=1, kernel_size=11):
        super().__init__()
        # in_channels=2 represents I and Q (Real and Imaginary) components
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        return self.network(x)
```

## 3. The Neural DSP Block Wrapper

To integrate this PyTorch model into our existing pipeline, we wrap it in a `DSPBlock`. This block handles the conversion between `SignalBuffer` and PyTorch Tensors.

```python
import numpy as np
from typing import Any
from dsp_block_api import DSPBlock
from signal_buffer_spec import SignalBuffer

class NeuralEqualizerBlock(DSPBlock):
    """
    DSP Block that wraps a PyTorch neural network for non-linear equalization.
    """
    def __init__(self, pytorch_model: nn.Module, model_weights_path: str, device: str = 'cpu') -> None:
        """
        pytorch_model: The uninitialized PyTorch model class instance.
        model_weights_path: Path to the .pth or .pt file containing trained weights.
        device: 'cpu' or 'cuda' for GPU acceleration.
        """
        super().__init__(model_weights_path=model_weights_path, device=device)
        self.device = torch.device(device)
        
        # Load Model
        self.model = pytorch_model.to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval() # Set to inference mode
        
        self.reset()

    def reset(self) -> None:
        """Clear historical overlap buffer if implementing overlap-save streaming."""
        self.overlap_buffer = None

    def process(self, input_buf: SignalBuffer, **kwargs: Any) -> SignalBuffer:
        data = input_buf.data
        
        # 1. Format NumPy array to PyTorch Tensor
        # If complex, split into two channels (Real, Imag)
        if np.iscomplexobj(data):
            tensor_data = np.vstack((data.real, data.imag))
        else:
            tensor_data = np.expand_dims(data, axis=0)
            
        # PyTorch expects shape: (Batch_Size, Channels, Sequence_Length)
        # We process 1 chunk as 1 batch: (1, 2, N)
        input_tensor = torch.from_numpy(tensor_data).float().unsqueeze(0).to(self.device)
        
        # 2. Run Neural Network Inference
        with torch.no_grad(): # Disable gradient calculation for faster inference
            output_tensor = self.model(input_tensor)
            
        # 3. Format back to NumPy
        # Remove batch dimension and move to CPU
        out_numpy = output_tensor.squeeze(0).cpu().numpy()
        
        # Reconstruct complex signal if necessary
        if np.iscomplexobj(data) and out_numpy.shape[0] == 2:
            final_out = out_numpy[0, :] + 1j * out_numpy[1, :]
        elif out_numpy.shape[0] == 1:
            final_out = out_numpy[0, :]
        else:
            final_out = out_numpy
            
        return input_buf.clone_metadata(final_out)
```

## 4. Addressing the Recurrent Neural Network (RNN/LSTM) Alternative

While CNNs handle local time-invariant dispersion well, they struggle with "long memory" effects (e.g., thermal droop in lasers or long cable reflections). 

**RNNs / LSTMs** are used when the current equalized symbol depends heavily on a long history of previous states. 

* **The Challenge:** RNNs process data sequentially. A standard PyTorch LSTM cannot be easily parallelized across the time dimension, making it substantially slower than a CNN for high-baud-rate DSP.
* **The Solution:** In modern DSP research, pure RNNs are being phased out in favor of **Transformers (Self-Attention)** or highly optimized **Temporal Convolutional Networks (TCNs)** with dilated convolutions, which mimic long-memory behavior but can be computed entirely in parallel on GPUs.

## 5. Pipeline Integration Example

```python
# 1. Define the architecture and load pre-trained weights
# (Assuming 'cnn_equalizer_weights.pth' was trained offline using a large dataset)
nn_eq_model = CNN1DEqualizerModel()
neural_eq_block = NeuralEqualizerBlock(
    pytorch_model=nn_eq_model,
    model_weights_path="models/cnn_equalizer_weights.pth",
    device="cuda" # Exploit GPU if available for massive array processing
)

# 2. Insert into the standard pipeline
# No changes to the pipeline logic are required!
advanced_system = DSPPipeline([
    CrosstalkChannelBlock(thru, next_taps), 
    neural_eq_block, 
    SymbolDemapperBlock(constellation)
])

rx_symbols = advanced_system.process(tx_signal)
```