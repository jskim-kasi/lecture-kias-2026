# GPU Computing & CUDA Programming Lecture — KIAS 2026

This repository contains the lecture materials, coding examples, benchmarks, and homework assignments for the GPU computing and CUDA programming course given by **Jongsoo Kim** at KIAS (Korea Institute for Advanced Study) in 2026.

## Repository Structure

The repository is organized into distinct directories for source codes, homework, slides, and conceptual documentation:

```
.
├── LICENSE                    # MIT License
├── README.md                  # Root overview (this file)
├── docs/                      # Theoretical documentation & deep-dive analysis
│   └── matrix_multiplication.md # Detailed V100/H100 specs, Roofline Model, and analysis
├── codes/                     # GPU benchmark codes, scripts, and results
│   ├── README.md              # Reorganized running instructions and performance results
│   ├── vector_add/            # Introductory GPU vector addition examples
│   │   ├── vector_add_cutile.py # cuTile implementation (run on H100)
│   │   └── vector_add_numba.py  # Numba implementation (run on V100)
│   └── matrix_mul/            # Matrix multiplication scripts, results, and roofline benchmarks
│       ├── matrix_mul_cutile.py # cuTile implementation (run on H100)
│       ├── matrix_mul_cutile_roofline.py # cuTile roofline profiling & plotting (run on H100)
│       ├── *.py               # Numba, CuPy, and nvmath matrix multiplication scripts
│       ├── *.csv              # Tabulated benchmark results
│       └── *.png              # Generated Roofline model plots
├── homework/                  # Student assignments and exercises
└── slides/                    # Lecture slides and presentation files
```

## Directory Overview

### 1. [docs/](docs/)
*   **[matrix_multiplication.md](docs/matrix_multiplication.md)**: An in-depth theoretical and architectural guide. Explains Arithmetic Intensity calculation, the Peak Compute and Memory Bandwidth limits of the NVIDIA V100 and H100 GPUs, custom tiling vs. cuBLAS templates, and nvmath epilog fusion.

### 2. [codes/](codes/)
*   **Vector Addition**: Simple 1D execution configurations using Numba CUDA (on V100) and NVIDIA's new cuTile DSL (on H100).
*   **Matrix Multiplication Benchmarks**: Performance comparison of floating-point matrix multiplication ($1024 \times 1024$) using CPU (NumPy), CuPy, Numba CUDA, and `nvmath-python` on V100, alongside NVIDIA's new cuTile implementation on H100.
*   **Roofline Model Analysis**: Measurements of operational intensity and throughput for matrix sizes ranging from 32 to 16,384, plotted against the theoretical V100 roofline limit.
*   **nvmath-python Epilog Fusion**: Performance demonstration fusing GEMM + Bias + ReLU using `cuBLASLt` epilogs.

### 3. [homework/](homework/)
*   Exercises and tasks assigned during the lecture.

### 4. [slides/](slides/)
*   Lecture notes, presentation slides, and reference materials.

---

*For detailed setup instructions, module configuration, and running the scripts, please refer to the [codes README](codes/README.md).*
