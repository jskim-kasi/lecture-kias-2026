# GPU Computing & CUDA Programming Lecture — KIAS 2026

This repository contains the lecture materials, coding examples, benchmarks, and homework assignments for the GPU computing and CUDA programming course given by **Jongsoo Kim** at KIAS (Korea Institute for Advanced Study) in 2026.

## Repository Structure

The repository is organized with a clean, flat directory layout:

```
.
├── LICENSE                    # MIT License
├── README.md                  # Root overview (this file)
├── codes/                     # Benchmark codes, scripts, plots, and analysis
│   ├── README.md              # Detailed running instructions and performance results
│   ├── explanation.md         # In-depth architectural & profiling explanation
│   ├── *.py                   # Numba, CuPy, and nvmath matrix multiplication scripts
│   ├── *.csv                  # Tabulated performance results
│   └── *.png                  # Roofline model plots
├── homework/                  # Student assignments and exercises
└── slides/                    # Lecture slides and presentation files
```

## Directory Overview

### 1. [codes/](codes/)
*   **Matrix Multiplication Benchmarks**: Performance comparison of floating-point matrix multiplication ($1024 \times 1024$) using CPU (NumPy), CuPy, Numba CUDA, and `nvmath-python`.
*   **Roofline Model Analysis**: Code to measure operational intensity and throughput for matrix sizes ranging from 32 to 16,384, plotting them against a V100 GPU's hardware limits.
*   **nvmath-python Epilog Fusion**: Demonstration of fusing matrix multiplication with bias addition and ReLU activation using NVIDIA's `nvmath-python` library.
*   **Documentation**: Includes execution guides and a deep-dive analysis comparing custom GPU kernel performance (e.g. Numba tiling) vs. vendor-optimized libraries (cuBLAS).

### 2. [homework/](homework/)
*   Exercises and tasks assigned during the lecture.

### 3. [slides/](slides/)
*   Lecture notes, presentation slides, and reference materials.

---

*For detailed instructions on setting up your environment, loading CUDA modules, and running/profiling the benchmark scripts, please refer to the [codes README](codes/README.md).*
