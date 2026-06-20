# GPU & CPU Matrix Multiplication Benchmarks

This project contains Python scripts to perform and profile matrix multiplication of size `1024x1024` with float32 precision, using CPU (NumPy), GPU (CuPy), and GPU (nvmath-python). It also contains a benchmarking script to plot the Roofline Model of the Tesla V100 GPU.

### File Structure & Hardware Setup

The benchmark scripts are split into two directories:
*   **Tesla V100 GPU SXM2 (32GB)**: Used for all matrix multiplication codes (CuPy, nvmath, Numba) and Numba vector addition.
*   **NVIDIA H100 GPU**: Used for the cuTile implementations (vector addition and matrix multiplication) utilizing CUDA 13.x features.

### 1. Vector Addition ([vector_add/](vector_add/))
- [vector_add_numba.py](vector_add/vector_add_numba.py) (V100): GPU script demonstrating vector addition using Numba's `@cuda.jit` compiler and memory management.
- [vector_add_cutile.py](vector_add/vector_add_cutile.py) (H100): GPU script demonstrating vector addition using NVIDIA's new cuTile Python DSL (`cuda.tile`) and block-level tile loading/storing.

### 2. Matrix Multiplication ([matrix_mul/](matrix_mul/))
- [matrix_mul_num.py](matrix_mul/matrix_mul_num.py): CPU script generating two random `1024x1024` float32 matrices and multiplying them using NumPy.
- [matrix_mul_cupy.py](matrix_mul/matrix_mul_cupy.py): GPU script generating two random `1024x1024` float32 matrices and multiplying them on the GPU using CuPy.
- [matrix_mul_nvmath.py](matrix_mul/matrix_mul_nvmath.py): GPU script using NVIDIA's [nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest/index.html) library wrapper with the stateful `Matmul` class context.
- [matrix_mul_numba.py](matrix_mul/matrix_mul_numba.py): GPU script demonstrating $1024 \times 1024$ matrix multiplication using Numba's `@cuda.jit` compiler and 2D execution configuration.
- [matrix_mul_cutile.py](matrix_mul/matrix_mul_cutile.py) (H100): GPU script performing matrix multiplication using NVIDIA's new cuTile Python DSL (`cuda.tile`) and 2D block-level tile multiply-accumulate (`ct.mma`).
- [matrix_mul_cutile_roofline.py](matrix_mul/matrix_mul_cutile_roofline.py) (H100): Benchmarking script that loops matrix sizes from 32 to 16384 using cuTile, saves measurements to [benchmark_cutile_results.csv](matrix_mul/benchmark_cutile_results.csv), and generates the Roofline plot [roofline_cutile.png](matrix_mul/roofline_cutile.png).
- [matrix_mul_cupy_roofline.py](matrix_mul/matrix_mul_cupy_roofline.py): Benchmarking script that loops matrix sizes from 32 to 16384 for CuPy, saves to [benchmark_cupy_results.csv](matrix_mul/benchmark_cupy_results.csv), and generates the plot [roofline_cupy.png](matrix_mul/roofline_cupy.png).
- [matrix_mul_nvmath_roofline.py](matrix_mul/matrix_mul_nvmath_roofline.py): Benchmarking script that loops matrix sizes from 32 to 16384 using stateful nvmath-python, saves measurements to [benchmark_nvmath_results.csv](matrix_mul/benchmark_nvmath_results.csv), and generates [roofline_nvmath.png](matrix_mul/roofline_nvmath.png).
- [matrix_mul_numba_roofline.py](matrix_mul/matrix_mul_numba_roofline.py): Benchmarking script that loops matrix sizes from 32 to 16384 using Numba shared-memory tiling, saves measurements to [benchmark_numba_results.csv](matrix_mul/benchmark_numba_results.csv), and generates [roofline_numba.png](matrix_mul/roofline_numba.png).
- [plot_comparison_roofline.py](matrix_mul/plot_comparison_roofline.py): Unified benchmarking and comparison plotting script that runs benchmarks across sizes 32 to 16384 for all three backends, exports three CSVs, and generates the combined comparison plot [roofline_comparison.png](matrix_mul/roofline_comparison.png).
- [matrix_mul_nvmath_fusion.py](matrix_mul/matrix_mul_nvmath_fusion.py): Demonstrates forward-only epilog fusion (GEMM + Bias + ReLU) comparing inlined naive execution vs. `RELU_BIAS` epilog.

---

## Prerequisites

1. Load the python environment module:
   ```bash
   module load python/3.12.12
   ```
   *(Note: We use `python/3.12.12` because the default `python/3.14.0` preloads CUDA 13.x pip packages globally, which conflict with Volta-architecture GPUs like the Tesla V100)*
2. Install `cupy-cuda12x` and `nvmath-python[cu12-dx]==0.9.0` user-wide:
   ```bash
   pip install --user cupy-cuda12x nvmath-python[cu12-dx]==0.9.0
   ```
3. Set the CUDA library path:
   ```bash
   export LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
   ```

---

## Running the Code

### 1. Run CPU Version (NumPy)
```bash
module load python/3.12.12
python3 codes/matrix_mul/matrix_mul_num.py
```

### 2. Run GPU Version (CuPy)
```bash
module load python/3.12.12
LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" python3 codes/matrix_mul/matrix_mul_cupy.py
```

### 3. Run GPU Version (nvmath-python Stateful)
```bash
module load python/3.12.12
LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" python3 codes/matrix_mul/matrix_mul_nvmath.py
```

#### Expected nvmath-python Output:
```
Initializing two 1024x1024 matrices on the GPU (using CuPy & nvmath)...
Planning GEMM execution...
Warming up nvmath-python GEMM...
Performing matrix multiplication A @ B using stateful nvmath-python...

---------------- Results ----------------
Result shape: (1024, 1024)
Result dtype: float32
Avg Processing time: 0.000412 seconds
Performance: 5210.83 GFLOPs
Total FLOPs: 2,147,483,648 FLOPs
-----------------------------------------
```

### 4. Run Combined Benchmark & Generate Roofline Comparison Plot
To run the unified benchmark for all three backends (CuPy, nvmath-python, and Numba CUDA) and generate the comparison plot:
```bash
module load python/3.12.12
LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" python3 codes/matrix_mul/plot_comparison_roofline.py
```
*(Note: If you run this script, it will export results and save `roofline_comparison.png` into `codes/matrix_mul/`.)*

### 5. Run Forward Epilog Fusion Demo
To run the epilog fusion code:
```bash
module load python/3.12.12
LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" python3 codes/matrix_mul/matrix_mul_nvmath_fusion.py
```

### 6. Run Numba Vector Addition (V100)
To run the Numba CUDA vector addition example:
```bash
module load python/3.12.12
LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" python3 codes/vector_add/vector_add_numba.py
```

### 7. Run Numba Matrix Multiplication (V100)
To run the Numba CUDA matrix multiplication example:
```bash
module load python/3.12.12
LD_LIBRARY_PATH="/opt/ohpc/pub/cuda/12.8.1/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH" python3 codes/matrix_mul/matrix_mul_numba.py
```

### 8. Run cuTile Vector Addition (H100)
To run the cuTile vector addition example (using a local environment supporting the `cuda.tile` module on the H100 GPU):
```bash
python3 codes/vector_add/vector_add_cutile.py
```

### 9. Run cuTile Matrix Multiplication (H100)
To run the cuTile matrix multiplication example (which compiles and executes utilizing Hopper Tensor Core MMA instructions):
```bash
python3 codes/matrix_mul/matrix_mul_cutile.py
```

### 10. Run cuTile Matrix Multiplication Roofline (H100)
To run the cuTile matrix multiplication benchmarking loop and generate the Roofline plot:
```bash
python3 codes/matrix_mul/matrix_mul_cutile_roofline.py
```
*(Note: If you run this script, it will export results and save `roofline_cutile.png` into `codes/matrix_mul/`.)*

---

## Roofline Benchmark Results (Tesla V100 SXM2)

- **Peak FP32 performance**: 15.7 TFLOPS (15,700 GFLOPS)
- **Peak Memory Bandwidth**: 900 GB/s

| Matrix Size | AI (FLOP/B) | CuPy Time (s) | CuPy (GFLOPS) | nvmath Time (s) | nvmath (GFLOPS) | Numba Time (s) | Numba (GFLOPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **32x32** | 5.33 | 0.000064 | 1.02 | 0.000099 | 0.66 | 0.000089 | 0.73 |
| **64x64** | 10.67 | 0.000039 | 13.43 | 0.000085 | 6.20 | 0.000078 | 6.75 |
| **128x128** | 21.33 | 0.000045 | 93.85 | 0.000099 | 42.41 | 0.000075 | 56.30 |
| **256x256** | 42.67 | 0.000063 | 534.04 | 0.000194 | 173.21 | 0.000086 | 391.58 |
| **512x512** | 85.33 | 0.000075 | 3,578.28 | 0.000173 | 1,551.16 | 0.000196 | 1,371.68 |
| **1024x1024** | 170.67 | 0.000298 | 7,203.27 | 0.000343 | 6,262.62 | 0.000973 | 2,206.16 |
| **2048x2048** | 341.33 | 0.001474 | 11,655.65 | 0.001495 | 11,491.63 | 0.007410 | 2,318.36 |
| **4096x4096** | 682.67 | 0.011038 | 12,451.78 | 0.011091 | 12,391.46 | 0.056573 | 2,429.41 |
| **8192x8192** | 1365.33 | 0.078239 | **14,053.20** | 0.078535 | **14,000.23** | 0.463398 | 2,372.72 |
| **16384x16384** | 2730.67 | 0.627137 | **14,025.80** | 0.627438 | **14,019.07** | 3.674601 | 2,393.75 |

CuPy results are exported to [benchmark_cupy_results.csv](matrix_mul/benchmark_cupy_results.csv) and plotted in [roofline_cupy.png](matrix_mul/roofline_cupy.png).
nvmath results are exported to [benchmark_nvmath_results.csv](matrix_mul/benchmark_nvmath_results.csv) and plotted in [roofline_nvmath.png](matrix_mul/roofline_nvmath.png).
Numba results are exported to [benchmark_numba_results.csv](matrix_mul/benchmark_numba_results.csv) and plotted in [roofline_numba.png](matrix_mul/roofline_numba.png).

---

## cuTile Benchmark Results (NVIDIA H100 PCIe)

- **Peak FP32 performance**: 51 TFLOPS (51,000 GFLOPS)
- **Peak Memory Bandwidth**: 2.0 TB/s (2,000 GB/s)

| Matrix Size | AI (FLOP/B) | cuTile Time (s) | cuTile (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **32x32** | 5.33 | 0.000025 | 2.66 |
| **64x64** | 10.67 | 0.000015 | 35.34 |
| **128x128** | 21.33 | 0.000019 | 222.72 |
| **256x256** | 42.67 | 0.000058 | 577.09 |
| **512x512** | 85.33 | 0.000315 | 852.97 |
| **1024x1024** | 170.67 | 0.002356 | 911.42 |
| **2048x2048** | 341.33 | 0.018596 | 923.86 |
| **4096x4096** | 682.67 | 0.136846 | 1,004.33 |
| **8192x8192** | 1365.33 | 1.088350 | 1,010.26 |
| **16384x16384** | 2730.67 | 8.667315 | **1,014.86** |

*   **vector_add_cutile.py** (vector size `1,024,000`): Executes in **0.078 ms**.

cuTile results are exported to [benchmark_cutile_results.csv](matrix_mul/benchmark_cutile_results.csv) and plotted in [roofline_cutile.png](matrix_mul/roofline_cutile.png).
