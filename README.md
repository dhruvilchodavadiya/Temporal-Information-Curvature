# Temporal Information Curvature (TIC)
*A robust, time-aware diagnostic for training instability in machine learning and other temporal signals.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#installation)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#tests)

---

## What is TIC?
**Temporal Information Curvature (TIC)** is an operator on a time series \( f(t) \) that detects early-onset instability by combining curvature, nonlinear feedback, and memory:

\[
\mathrm{TIC}[f](t) \;=\; \frac{d^2 f}{dt^2} \;-\; \gamma \left(\frac{df}{dt}\right)^2 \;+\; \phi(t)\, f(t),
\]

- \( \gamma \) — feedback sensitivity (damps large gradients)
- \( \phi(t) \) — memory kernel (weights historical influence)

TIC fires **early**, is **robust to noise**, and yields a **single actionable signal** you can wire into training loops (halt, lower LR, gradient clip, etc.).

---

## Features
- **Plug-and-play**: one function call to compute TIC on any 1D time series.
- **Noise-robust**: nonlinear feedback suppresses jitter from stochastic training.
- **Actionable**: built-in decision logic with threshold + minimum duration.
- **Framework-friendly**: examples for PyTorch; easy to extend to TF/JAX.
- **General**: works for ML loss, volatility series, EEG/audio, etc.

---
