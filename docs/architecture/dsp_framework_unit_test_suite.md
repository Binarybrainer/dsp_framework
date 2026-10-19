# 🧪 DSP Framework Comprehensive Unit Test Suite

To maintain a production-ready framework, every `DSPBlock` must be validated. We use the standard Python `pytest` library alongside `numpy.testing` to mathematically verify the outputs. 

This document provides the test blueprints for the core infrastructure and all major DSP modules.

## 1. Core Infrastructure Tests

These tests ensure the base classes handle memory and metadata correctly.

```python
import pytest
import numpy as np
from dsp_framework.core.signal_buffer_spec import SignalBuffer
from dsp_framework.core.dsp_block_api import DSPPipeline

def test_signal_buffer_metadata():
    """Test that SignalBuffer correctly stores and clones metadata."""
    data = np.ones(100)
    buf = SignalBuffer(data, sample_rate=1e6, timestamp=0.5)
    
    assert buf.active_length == 100
    assert buf.sample_rate == 1e6
    
    # Test cloning
    new_data = np.zeros(100)
    cloned_buf = buf.clone_metadata(new_data)
    assert cloned_buf.timestamp == 0.5
    assert np.array_equal(cloned_buf.data, new_data)

def test_pipeline_chaining():
    """Test that DSPPipeline correctly passes data between multiple blocks."""
    from dsp_framework.blocks.channel_modeling_spec import AWGNChannel
    
    buf = SignalBuffer(np.zeros(1000), sample_rate=1e6)
    pipeline = DSPPipeline([AWGNChannel(snr_db=10), AWGNChannel(snr_db=10)])
    
    out_buf = pipeline.process(buf)
    assert out_buf.active_length == 1000
```

## 2. Signal Generation & Filtering Tests

These tests verify basic signal processing math and continuity.

```python
from dsp_framework.blocks.filters_resampling_spec import LTI_FilterBlock, FractionalResamplerBlock
from numpy.testing import assert_allclose

def test_fir_filter_dc_gain():
    """Test that a moving average filter correctly scales a DC signal."""
    # 4-tap moving average (DC gain = 1.0)
    taps = np.array([0.25, 0.25, 0.25, 0.25])
    filter_block = LTI_FilterBlock(b=taps, a=[1.0])
    
    buf = SignalBuffer(np.ones(100), sample_rate=1.0)
    out_buf = filter_block.process(buf)
    
    # After the initial 3 samples (transient), the output should be exactly 1.0
    assert_allclose(out_buf.data[3:], 1.0, atol=1e-7)

def test_fractional_resampler_ratio():
    """Test that the resampler produces the correct number of samples."""
    fs_in = 100.0
    fs_out = 150.0
    resampler = FractionalResamplerBlock(target_fs=fs_out)
    
    buf_in = SignalBuffer(np.random.randn(100), sample_rate=fs_in)
    buf_out = resampler.process(buf_in)
    
    # 100 samples at 100Hz = 1 second. 1 second at 150Hz = 150 samples.
    assert buf_out.active_length == 150
    assert buf_out.sample_rate == fs_out
```

## 3. Modulation & Demodulation Tests

Verify that mapping and demapping are perfect inverses of each other in a noiseless environment.

```python
from dsp_framework.blocks.modulation_demodulation_spec import SymbolMapperBlock, SymbolDemapperBlock

def test_qam16_back_to_back():
    """Test that Mapper -> Demapper yields 0% Bit Error Rate without noise."""
    M = 16
    tx_symbols = np.random.randint(0, M, 10000)
    tx_buf = SignalBuffer(tx_symbols, sample_rate=1e6)
    
    mapper = SymbolMapperBlock(modulation_type='QAM', M=M)
    demapper = SymbolDemapperBlock(constellation=mapper.constellation, decision_type='hard')
    
    iq_buf = mapper.process(tx_buf)
    rx_buf = demapper.process(iq_buf)
    
    # Check for perfect recovery
    assert np.array_equal(tx_buf.data, rx_buf.data)
```

## 4. Channel & Impairment Tests

Verify statistical properties of noise and channel models.

```python
from dsp_framework.blocks.channel_modeling_spec import AWGNChannel

def test_awgn_variance():
    """Test that the AWGN block injects the correct amount of noise power."""
    # Generate a signal with exactly 1.0 average power
    data = np.ones(100000, dtype=np.complex128)
    buf = SignalBuffer(data, sample_rate=1e6)
    
    snr_db = 10.0 # SNR = 10 means Signal Power is 10x Noise Power
    channel = AWGNChannel(snr_db=snr_db)
    
    out_buf = channel.process(buf)
    
    noise = out_buf.data - buf.data
    noise_power = np.mean(np.abs(noise)**2)
    
    # Expected noise power = 1.0 / 10^(10/10) = 0.1
    assert_allclose(noise_power, 0.1, rtol=0.05) # 5% tolerance due to randomness
```

## 5. Stateful & Feedback Loop Tests (PLL / CDR)

Verify that feedback loops converge correctly and maintain state across chunks.

```python
from dsp_framework.blocks.pll_specification import PLLBlock

def test_pll_phase_locking():
    """Test that the PLL successfully de-rotates a constant frequency offset."""
    N = 10000
    fs = 1e6
    f_offset = 5e3 # 5 kHz offset
    
    # Generate QPSK signal with frequency offset
    t = np.arange(N) / fs
    clean_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], N)
    rotated_symbols = clean_symbols * np.exp(1j * 2 * np.pi * f_offset * t)
    
    buf = SignalBuffer(rotated_symbols, sample_rate=fs)
    
    pll = PLLBlock(kp=0.1, ki=0.005)
    out_buf = pll.process(buf)
    
    # The first few samples will be garbage as the PLL locks
    lock_time = 2000 
    
    # After locking, the output should closely match the clean symbols
    error = np.abs(out_buf.data[lock_time:] - clean_symbols[lock_time:])
    assert np.mean(error) < 0.1 
```

## 6. C++ Parity Tests

Ensures the targeted C++ migration (Phase 4) does not break functionality.

```python
from dsp_framework.blocks.fast_pll import FastPLLBlock
# (Assuming the parity test from targeted_migration_spec.md is placed here)

def test_cpp_python_pll_parity():
    """Test that C++ PLL and Python PLL produce mathematically identical outputs."""
    # Initialization and processing as defined in targeted_migration_spec.md
    pass
```