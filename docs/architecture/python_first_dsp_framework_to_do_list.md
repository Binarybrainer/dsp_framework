# 🐍 Python-First DSP Framework Development To-Do List

This roadmap adopts a **Python-First / Rapid Prototyping** strategy. The goal is to build, verify, and visualize all DSP features using pure Python (NumPy/SciPy) first. Once the mathematical models and system architectures are proven, we will profile the code and migrate only the performance-critical bottlenecks to a C++ backend.

## Phase 1: Pure Python Foundations
*Focus: Setting up the core architecture, data structures, and basic utilities using NumPy and Matplotlib.*

- [x] **API Design & Core Architecture**
  - [x] Define the abstract base class `DSPBlock` with standard methods (`process()`, `reset()`, `update_params()`).
  - [x] Implement a `Pipeline` or `System` class to chain multiple `DSPBlock` instances together.
- [x] **Data Structures**
  - [x] Implement `SignalBuffer` wrapper around `numpy.ndarray` to track sampling rate, timestamp, and active data length.
- [x] **Signal Generators**
  - [x] Implement PRBS (Pseudo-Random Binary Sequence) generator.
  - [x] Implement basic waveforms (Sine, Cosine, Square wave generators).
- [x] **Visualization (Matplotlib Wrappers)**
  - [x] Implement Time-Domain plotting utility.
  - [x] Implement Frequency-Domain (FFT / PSD) plotting utility.
  - [x] Implement Constellation diagram scatter plot utility.

## Phase 2: Core DSP Blocks
*Focus: Essential signal processing algorithms implemented via SciPy and NumPy.*

- [x] **Modulation & Demodulation**
  - [x] Implement Symbol Mapper (BPSK, QPSK, M-QAM).
  - [x] Implement Symbol Demapper (Hard/Soft decision decoding).
- [ ] **Filters & Resampling**
  - [ ] Implement FIR and IIR filters leveraging `scipy.signal.lfilter`.
  - [ ] Implement Root Raised Cosine (RRC) pulse shaping filter.
  - [ ] Implement fractional resampler using SciPy interpolation.
- [ ] **Channel Modeling**
  - [ ] Add Additive White Gaussian Noise (AWGN) block.
  - [ ] Add Phase Noise and Frequency Offset impairment blocks.
  - [ ] Implement a basic Multipath Fading channel (Rayleigh/Rician).

## Phase 3: Advanced Systems (Rapid Prototyping)
*Focus: Complex control loops and multi-antenna systems built in pure Python to verify logic.*

- [ ] **Phase-Locked Loop (PLL)**
  - [ ] Implement Phase Frequency Detector (PFD).
  - [ ] Implement Digital Loop Filter (PI controller).
  - [ ] Implement VCO/NCO (Numerically Controlled Oscillator).
- [ ] **Clock and Data Recovery (CDR)**
  - [ ] Implement Alexander (Bang-Bang) Phase Detector.
  - [ ] Implement Phase Interpolator (PI) logic.
  - [ ] Create Eye Diagram visualization utility for CDR analysis.
- [ ] **MIMO Processing**
  - [ ] Implement basic $H$ Matrix channel modeling (e.g., $2 \times 2$ or $4 \times 4$).
  - [ ] Implement Zero-Forcing (ZF) and MMSE linear equalizers.

## Phase 4: Profiling & Backend Migration
*Focus: Identifying performance bottlenecks and seamlessly swapping them out with C++ extensions.*

- [ ] **Performance Profiling**
  - [ ] Integrate `cProfile` and `line_profiler` to analyze pipeline execution times.
  - [ ] Identify the top 3 slowest blocks (e.g., sample-by-sample feedback loops in CDR/PLL, or heavy convolutions).
- [ ] **C++ Backend Infrastructure**
  - [ ] Set up `pybind11` and CMake within the Python package structure.
  - [ ] Create a unified memory interface allowing zero-copy NumPy array passing to C++.
- [ ] **Targeted Migration**
  - [ ] Rewrite identified bottleneck blocks in C++20.
  - [ ] Wrap C++ blocks with `pybind11` so they perfectly match the `DSPBlock` Python API.
  - [ ] Run numerical parity tests to ensure the C++ backend output exactly matches the pure Python prototype.