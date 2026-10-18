# 🎯 Consolidated Task Roadmap: Easy to Hard

This document synthesizes all framework development tasks into an ordered progression—from foundational setup to advanced multi-language and DSP core engineering.

---

## 🟢 Level 1: Beginner / Foundational (Setup & File Structure)

Tasks requiring basic repository setup, file organization, and environment configuration.

1. **Repository & Directory Initialization**
   - Create standard directory hierarchy (`include/dsp/`, `src/`, `python/`, `julia/`, `tests/`, `examples/`).
   - Add `.gitignore`, `LICENSE`, and `README.md`.
2. **Build Configuration Basics**
   - Write basic root `CMakeLists.txt` targeting C++20.
   - Configure basic `pyproject.toml` for Python packaging.
   - Configure `Project.toml` for Julia package tracking.
3. **Basic C-ABI Descriptor Header**
   - Write `dsp_abi.h` defining memory layout structs (`dsp_signal_buffer_abi_t`, `dsp_frame_tensor_abi_t`) and standard enums (`dsp_datatype_t`).

---

## 🟡 Level 2: Intermediate (Core Data Models & Basic DSP)

Tasks involving fundamental DSP math, aligned memory buffers, and initial unit testing.

4. **Aligned Memory & Buffer Allocator**
   - Implement `SignalBuffer<T>` with 64-byte alignment (`posix_memalign` / `_aligned_malloc`).
   - Add RAII lifecycle management and move-semantics support.
5. **Basic Signal Generators & Filters**
   - Implement basic PRBS (Pseudo-Random Binary Sequence) generator.
   - Implement QAM / QPSK modulators.
   - Implement basic FIR/IIR filter routines and Raised Cosine (RRC) filtering.
6. **Basic C++ Unit Tests**
   - Set up GoogleTest / Catch2 in `tests/cpp/`.
   - Write unit tests for memory allocation, buffer strides, and basic filter responses.

---

## 🟠 Level 3: Advanced (Zero-Copy Interoperability & Complex DSP Modules)

Tasks covering cross-language bindings, zero-copy memory wrappers, and non-linear control loops.

7. **Zero-Copy Python Bindings (`pybind11` / `nanobind`)**
   - Expose `SignalBuffer` to Python using Python Buffer Protocol (`py::array_t`).
   - Implement reference-counting retention to prevent double-free bugs across Python runtime.
8. **Phase-Locked Loop (PLL) Engine**
   - Implement Phase Frequency Detector (PFD).
   - Implement 2nd/3rd order Digital Loop Filter (PI controller).
   - Implement VCO/DCO with phase noise injection.
9. **Clock and Data Recovery (CDR) Engine**
   - Implement Alexander (Bang-Bang) Phase Detector or Mueller-Müller PD.
   - Implement Phase Interpolator (PI) models and loop control logic.
10. **Zero-Copy Julia Bindings (`unsafe_wrap`)**
    - Expose C-ABI descriptors to Julia via CFFI (`ccall`).
    - Wrap native buffers using `unsafe_wrap(Vector, ptr; own=false)`.

---

## 🔴 Level 4: Expert / Complex (MIMO, Advanced Metrics & Cross-Lang Parity)

High-complexity tasks involving multi-antenna matrix algebra, cross-language numerical verification, and full-system simulations.

11. **MIMO Channel & Spatial Processing**
    - Implement Rayleigh/Rician fading matrix channel models $H$ with spatial correlation.
    - Implement SVD Precoding and Linear Equalizers (Zero-Forcing, MMSE).
    - Implement non-linear MIMO detectors (Sphere Decoding, Successive Interference Cancellation).
12. **Cross-Language Numerical Parity & Benchmark Suite**
    - Create cross-language test harness (`tests/parity/`) verifying that Python, C++, and Julia produce identical outputs within $10^{-6}$ error margin.
    - Benchmark memory overhead and throughput under high streaming rates.
13. **End-to-End Simulation Pipelines & Visualizations**
    - Build High-Speed SerDes Link example (PRBS $\rightarrow$ Jitter Channel $\rightarrow$ CDR $\rightarrow$ Eye Diagram).
    - Build $4 \times 4$ MIMO-OFDM simulation pipeline with BER vs. SNR evaluation curves.