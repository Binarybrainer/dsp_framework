# 💯 Comprehensive DSP Simulation Checklist (100 Exercises)

This checklist spans the entire spectrum of Digital Signal Processing, organized logically from fundamental operations to advanced post-silicon research.

## Phase 1: Signals, Systems & Math Basics
- [x] 1. Generate continuous Sine, Cosine, and Square waves.
- [ ] 2. Implement and plot Fast Fourier Transform (FFT) of a mixed-frequency signal.
- [ ] 3. Compute and plot Power Spectral Density (PSD) using Welch's method.
- [ ] 4. Perform discrete convolution in the time domain vs. frequency domain.
- [ ] 5. Calculate and plot Autocorrelation and Cross-correlation.
- [ ] 6. Generate True AWGN (Additive White Gaussian Noise) for a specific variance.
- [ ] 7. Simulate an ideal ADC (Quantization and Clipping).
- [ ] 8. Calculate Signal-to-Noise Ratio (SNR) and Signal-to-Quantization-Noise Ratio (SQNR).
- [ ] 9. Implement complex baseband representation (I/Q components).
- [ ] 10. Simulate group delay and phase delay of a simple system.

## Phase 2: Modulation & Demodulation
- [ ] 11. BPSK Modulation and Hard-Decision Demapping.
- [ ] 12. QPSK Modulation with Gray coding.
- [ ] 13. 16-QAM Modulation and Demodulation.
- [ ] 14. 64-QAM Modulation and Demodulation.
- [ ] 15. 256-QAM Modulation and Demodulation.
- [ ] 16. PAM4 Modulation (for high-speed SerDes).
- [ ] 17. PAM8 Modulation.
- [ ] 18. FSK (Frequency Shift Keying) generation.
- [ ] 19. Basic OFDM (Orthogonal Frequency Division Multiplexing) symbol generation using IFFT.
- [ ] 20. Plot theoretical vs. simulated Symbol Error Rate (SER) curves for QAM over AWGN.

## Phase 3: Channel Impairments
- [ ] 21. Simulate a flat-fading Rayleigh channel.
- [ ] 22. Simulate a flat-fading Rician channel (with line-of-sight component).
- [ ] 23. Simulate Frequency-Selective Fading (Multipath delay profile).
- [ ] 24. Apply a deterministic Frequency Offset to a carrier.
- [ ] 25. Simulate Oscillator Phase Noise (Wiener process/Random Walk).
- [ ] 26. Simulate Non-linear amplifier distortion (AM-AM and AM-PM).
- [ ] 27. Implement a static Crosstalk channel (simple FIR interference).
- [ ] 28. Simulate deterministic jitter (DJ) and random jitter (RJ).
- [ ] 29. Simulate clipping and saturation in a receiver front-end.
- [ ] 30. Implement an S-Parameter (.s2p) based insertion loss channel.

## Phase 4: Filtering & Resampling
- [ ] 31. Design and apply a moving average filter.
- [ ] 32. Design a Low-Pass FIR filter using Windowing (Hamming/Blackman).
- [ ] 33. Design an IIR Butterworth filter and check stability.
- [ ] 34. Implement a Matched Filter (Root Raised Cosine) for pulse shaping.
- [ ] 35. Decimation: Downsample a signal with an anti-aliasing filter.
- [ ] 36. Interpolation: Upsample a signal with an anti-imaging filter.
- [ ] 37. Fractional Resampling (e.g., sample rate conversion by 3/4).
- [ ] 38. Implement a CIC (Cascaded Integrator-Comb) filter.
- [ ] 39. Measure the Step Response and Impulse Response of an unknown filter block.
- [ ] 40. Equalize filter group delay to maintain symbol phase.

## Phase 5: Synchronization (CDR & PLL)
- [ ] 41. Implement a basic Phase Detector (Multiplier + Lowpass).
- [ ] 42. Implement a Proportional-Integral (PI) Loop Filter.
- [ ] 43. Simulate a Numerically Controlled Oscillator (NCO).
- [ ] 44. Build a complete Costas Loop for QPSK carrier recovery.
- [ ] 45. Implement a Gardner Timing Error Detector.
- [ ] 46. Implement a Mueller-Muller (Baud-rate) Phase Detector.
- [ ] 47. Implement an Alexander (Bang-Bang) Phase Detector.
- [ ] 48. Build a Phase Interpolator (PI) based Clock and Data Recovery (CDR) loop.
- [ ] 49. Simulate Cycle Slips in a PLL under low SNR.
- [ ] 50. Frame Synchronization using Barker codes and Cross-correlation.

## Phase 6: Equalization
- [ ] 51. Zero-Forcing (ZF) Linear Equalization for a known channel.
- [ ] 52. Minimum Mean Square Error (MMSE) Linear Equalization.
- [ ] 53. Implement a Feed-Forward Equalizer (FFE) using FIR taps.
- [ ] 54. Implement a Decision Feedback Equalizer (DFE) to cancel post-cursor ISI.
- [ ] 55. Least Mean Squares (LMS) Adaptive Equalizer.
- [ ] 56. Recursive Least Squares (RLS) Adaptive Equalizer.
- [ ] 57. Constant Modulus Algorithm (CMA) for Blind Equalization.
- [ ] 58. Viterbi Algorithm (Maximum Likelihood Sequence Estimation - MLSE).
- [ ] 59. Continuous-Time Linear Equalizer (CTLE) analog approximation.
- [ ] 60. Combined FFE + DFE system for severe ISI channels.

## Phase 7: Multi-Antenna & MIMO
- [ ] 61. 2x2 MIMO Channel modeling (H Matrix).
- [ ] 62. 4x4 MIMO Channel modeling.
- [ ] 63. Alamouti Space-Time Block Coding (STBC) transmission and reception.
- [ ] 64. MIMO Zero-Forcing Receiver.
- [ ] 65. MIMO MMSE Receiver.
- [ ] 66. Singular Value Decomposition (SVD) for MIMO precoding.
- [ ] 67. Water-filling algorithm for MIMO power allocation.
- [ ] 68. Massive MIMO (e.g., 64x16) array generation and beamforming.
- [ ] 69. OFDM-MIMO System (combining spatial multiplexing with subcarriers).
- [ ] 70. Channel Estimation using Pilot Symbols (Least Squares).

## Phase 8: High-Speed SerDes & Signal Integrity
- [ ] 71. Generate and plot a "Closed" Eye Diagram for 56G PAM4.
- [ ] 72. Generate and plot an "Open" Eye Diagram post-equalization.
- [ ] 73. Extract THRU, NEXT, and FEXT impulse responses from a multi-port .s8p file.
- [ ] 74. Simulate a victim lane with 3 active NEXT aggressors.
- [ ] 75. Simulate a victim lane with 3 active FEXT aggressors.
- [ ] 76. Time-Interleaved ADC (TI-ADC) with offset mismatch.
- [ ] 77. TI-ADC with gain mismatch.
- [ ] 78. TI-ADC with timing skew (sample phase offset).
- [ ] 79. Blind calibration algorithm for TI-ADC offset mismatch.
- [ ] 80. Simulate 112G PAM4 transmission through 30dB loss channel + DFE recovery.

## Phase 9: Optical & Co-Packaged Optics (CPO)
- [ ] 81. Simulate CW Laser output with Relative Intensity Noise (RIN).
- [ ] 82. Mach-Zehnder Modulator (MZM) cosine-squared non-linear transfer function.
- [ ] 83. Simulate Single-Mode Fiber (SMF) attenuation over distance.
- [ ] 84. Simulate Chromatic Dispersion (CD) in optical fiber.
- [ ] 85. Photodiode Opto-electrical conversion (Responsivity).
- [ ] 86. Transimpedance Amplifier (TIA) thermal noise modeling.
- [ ] 87. Signal-dependent Shot Noise modeling at the receiver.
- [ ] 88. Coherent Optical Detection (Local Oscillator mixing).
- [ ] 89. Phase tracking for coherent optical systems (Viterbi-Viterbi algorithm).
- [ ] 90. End-to-end IM/DD CPO Link Power Budget simulation.

## Phase 10: AI/ML & Next-Gen DSP
- [ ] 91. Dataset generation: Export noisy/distorted signal buffers to PyTorch DataLoaders.
- [ ] 92. Train a 1D-CNN to replace a linear FFE.
- [ ] 93. Implement a PyTorch Neural Equalizer Block for inference in the DSP pipeline.
- [ ] 94. Train a Recurrent Neural Network (LSTM) to replace a DFE.
- [ ] 95. Train a Temporal Convolutional Network (TCN) for long-memory channel equalization.
- [ ] 96. Use a Multi-Layer Perceptron (MLP) for non-linear MZM compensation.
- [ ] 97. Autoencoder-based communication system (learning joint Tx/Rx mappings).
- [ ] 98. Reinforcement Learning applied to CDR Phase Interpolator control.
- [ ] 99. Implement an Extended Kalman Filter for joint phase/frequency tracking.
- [ ] 100. Simulate a basic Joint Radar and Communication (ISAC) OFDM waveform reflection.