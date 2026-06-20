# Matrix Multiplication & Vector Addition Explanations (CPU & GPU)

This document provides a detailed breakdown of the CPU-based [matrix_mul_num.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_num.py), the GPU-based [matrix_mul_cupy.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_cupy.py), the GPU-based [matrix_mul_nvmath.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_nvmath.py), the benchmark/plotting scripts [matrix_mul_cupy_roofline.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_cupy_roofline.py), [matrix_mul_nvmath_roofline.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_nvmath_roofline.py), and [matrix_mul_numba_roofline.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_numba_roofline.py), the unified benchmarking and comparison plotting script [plot_comparison_roofline.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/plot_comparison_roofline.py), the forward epilog fusion script [matrix_mul_nvmath_fusion.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_nvmath_fusion.py), the Numba CUDA vector addition script [vector_add_numba.py](file:///home/jskim/git/lecture-kias-2026/codes/vector_add/vector_add_numba.py), the Numba CUDA matrix multiplication script [matrix_mul_numba.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_numba.py), the cuTile vector addition script [vector_add_cutile.py](file:///home/jskim/git/lecture-kias-2026/codes/vector_add/vector_add_cutile.py), the cuTile matrix multiplication script [matrix_mul_cutile.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_cutile.py), and the cuTile roofline profiling script [matrix_mul_cutile_roofline.py](file:///home/jskim/git/lecture-kias-2026/codes/matrix_mul/matrix_mul_cutile_roofline.py) implementations.

---

## 1. CPU Version: `matrix_mul_num.py`

Runs on the host CPU using **NumPy**'s optimized linear algebra backend.

### Key Logic
1. **Imports**: Uses Python's standard `time` module and `numpy`.
2. **Matrix Initialization**: Allocates two random $1024 \times 1024$ float32 matrices (each requiring 4MB memory).
3. **Multiplication**: Computes `C = A @ B`. Under the hood, NumPy offloads this to BLAS (like OpenBLAS or MKL) using cache-blocked algorithms and SIMD (AVX/SSE) multithreading.

---

## 2. GPU Version: `matrix_mul_cupy.py`

Runs on the device GPU (Tesla V100) using **CuPy**'s `cuBLAS` library.

### Key Logic
1. **Imports**: Uses `cupy` (CuPy mimics NumPy's API but targets CUDA).
2. **Matrix Allocation**:
   ```python
   A = cp.asarray(np.random.rand(size, size).astype(np.float32))
   ```
   Generates a matrix on the host and transfers it to device (GPU) memory. (We use this method because host-device copy bypasses the `CURAND` driver mismatch issues present in the environment).
3. **Warmup Region**:
   ```python
   _ = A @ B
   cp.cuda.Device(0).synchronize()
   ```
   The first matrix multiplication triggers initialization overhead (cuBLAS handle creation, kernel compile/JIT caching). We run a warmup pass to ensure we measure raw kernel performance in the timed run.
4. **Synchronization**:
   ```python
   cp.cuda.Device(0).synchronize()
   ```
   CUDA calls in Python are asynchronous; they enqueue kernels in command queues and return control to Python immediately. Synchronizing the device blocks the CPU thread until the GPU finishes all operations, ensuring accurate timing.

---

## 3. GPU Version (nvmath-python Stateful): `matrix_mul_nvmath.py`

Runs on the GPU (Tesla V100) using NVIDIA's **nvmath-python** library, utilizing its stateful `Matmul` class context to separate planning and compilation from actual execution.

### Key Logic
1. **Stateful Context**:
   ```python
   with nvmath.linalg.advanced.Matmul(A, B) as mm:
       mm.plan()
       C = mm.execute()
   ```
   - **`mm.plan()`**: Analyzes the shapes, qualifiers, and datatypes, query-ing `cuBLAS` for compatible algorithms and selecting the optimal run schedule.
   - **`mm.execute()`**: Launches the pre-planned kernel.
2. **Performance Impact**: By invoking `mm.plan()` outside the timing loop, we eliminate all planning, library dispatching, and JIT compilation check overheads from the measured benchmark. This results in a massive speedup (increasing performance from **1,831 GFLOPs** in the stateless version to **5,210 GFLOPs** in the stateful version for size $1024 \times 1024$).

---

## 4. Mathematical & Profiling Metrics

### A. Arithmetic Intensity (AI)
Arithmetic Intensity measures how many floating-point operations (FLOPs) are performed per byte of memory read/written:
$$\text{Arithmetic Intensity (AI)} = \frac{\text{Floating-Point Operations (FLOPs)}}{\text{Memory Traffic (Bytes)}}$$

For a standard $N \times N$ matrix multiplication $C = A \times B$:
- **Total FLOPs**: Each cell in $C$ requires $N$ multiplications and $N$ additions.
  $$\text{FLOPs} = 2 \cdot N^3 = 2 \cdot 1024^3 = 2,147,483,648 \text{ FLOPs} \approx 2.15\text{ GigaFLOPs}$$
- **Memory Traffic (Bytes)**:
  - Read matrix $A$: $N^2$ elements
  - Read matrix $B$: $N^2$ elements
  - Write matrix $C$: $N^2$ elements
  - Total elements transferred = $3 \cdot N^2$ elements. Using 32-bit floats (4 bytes per element):
  $$\text{Bytes} = 3 \cdot N^2 \cdot 4 = 3 \cdot 1024^2 \cdot 4 = 12,582,912 \text{ Bytes} \approx 12.00 \text{ MB}$$

For $N = 1024$:
$$\text{AI} = \frac{2 \cdot N^3}{3 \cdot N^2 \cdot 4} = \frac{N}{6} = \frac{1024}{6} \approx 170.67 \text{ FLOP/Byte}$$

As the matrix size $N$ increases, the Arithmetic Intensity increases linearly with $N$ ($\text{AI} = N/6$). 

### B. Throughput Performance (GFLOPs)
Indicates how fast the calculation runs:
$$\text{GFLOPs} = \frac{\text{Total FLOPs}}{\text{Time (seconds)} \times 10^9}$$

- **NumPy CPU Speed (1024x1024)**: ~0.014760 seconds $\approx 145$ GFLOPs.
- **nvmath-python Stateless GPU Speed (1024x1024)**: ~0.001173 seconds $\approx 1,831$ GFLOPs.
- **nvmath-python Stateful GPU Speed (1024x1024)**: ~0.000343 seconds $\approx 6,263$ GFLOPs.
- **CuPy GPU Speed (1024x1024)**: ~0.000298 seconds $\approx 7,203$ GFLOPs.
- **CuPy GPU Speed (16384x16384)**: ~0.627137 seconds $\approx 14,026$ GFLOPs (14.03 TFLOPS).
- **nvmath-python Stateful GPU Speed (16384x16384)**: ~0.627438 seconds $\approx 14,019$ GFLOPs (14.02 TFLOPS).

---

## 5. The Roofline Model (Tesla V100 SXM2)

The Roofline Model visualizes the hardware constraints of a computer system:
- **Memory Bandwidth limit** (slope on the left): Bound by how fast data can be fetched from HBM2 memory (900 GB/s for Tesla V100).
- **Peak Compute performance limit** (flat ceiling): Bound by the raw processing power of the CUDA cores (15.7 TFLOPS for V100 standard single precision).

### Ridge Point
The transition point between memory-bound and compute-bound regimes is:
$$I_{\text{ridge}} = \frac{\text{Peak Compute}}{\text{Memory Bandwidth}} = \frac{15700 \text{ GFLOPS}}{900 \text{ GB/s}} \approx 17.44 \text{ FLOP/Byte}$$

- For computations with $\text{AI} < 17.44$ FLOP/Byte, performance is constrained by memory bandwidth.
- For computations with $\text{AI} \ge 17.44$ FLOP/Byte, performance is constrained by compute cores.

All our benchmarks ($N \ge 128$, $\text{AI} \ge 21.33$) lie in the compute-bound regime. For smaller sizes (like 32 to 256), the GPU suffers from latency overhead (kernel launch latency, host-device scheduling), meaning throughput remains low. As the matrix size increases (from 512 up to 16384), the compute capability of the GPU is fully saturated, and performance asymptotically approaches the 15.7 TFLOPS roofline, topping out at **14.05 TFLOPS** (89.51% efficiency) for $N=8192$.

---

## 6. Performance Comparison Analysis: CuPy vs. nvmath-python vs. Numba CUDA

### A. Observations & Data Trends
When comparing the performance points of CuPy, stateful `nvmath-python`, and Numba CUDA against the Tesla V100 Roofline model (as seen in `roofline_comparison.png`):
1. **Low-Intensity/Small Sizes ($N \le 512$):** Numba and CuPy exhibit high efficiency. For instance, at size $128 \times 128$, Numba achieves **56.30 GFLOPS** (with very low Python overhead) while CuPy achieves **93.85 GFLOPS**, outperforming `nvmath-python`'s **42.41 GFLOPS** which is dominated by high-level library dispatch latency.
2. **Mid-Intensity ($N = 1024$):** CuPy achieves **7,203 GFLOPS** (45.9% efficiency), nvmath achieves **6,263 GFLOPS** (39.9% efficiency), and Numba reaches **2,206 GFLOPS** (14.0% efficiency).
3. **High-Intensity ($N \ge 2048$):** The performance of CuPy and nvmath converges to **~14.02 TFLOPS** (89.3% efficiency) as the computation saturates the GPU cores. However, Numba's performance plateaues at **~2.4 TFLOPS**.

### B. Architectural & Optimization Differences
The differences in mid-to-high intensity throughput highlight the limits of custom compiler code compared to standard vendor linear algebra libraries:
- **cuBLAS Backend (CuPy & nvmath)**: Both CuPy and `nvmath-python` delegate the heavy matrix multiplication calculation to NVIDIA's high-performance **cuBLAS** C/C++ library. cuBLAS is highly optimized, containing assembly-tuned templates, register-level double-buffering, software pipelining, and hardware-specific instruction mapping (including Tensor Core execution path selection).
- **Custom Compiler Kernel (Numba)**: The Numba CUDA implementation uses a custom $16 \times 16$ tile shared-memory kernel. While shared-memory tiling reduces HBM global memory traffic drastically (preventing memory-bound stalls), Numba's generated PTX/SASS compiler code does not have the micro-architectural register blocking, software pipelining, or assembly tuning of cuBLAS, limiting its compute throughput to ~2.4 TFLOPS on the V100 GPU.

### C. The Latency-Throughput Trade-off
Python orchestration and dispatch overhead introduce a constant latency (in microseconds) per execution call.
- For **small matrices**, the GPU execution finishes almost instantly, making host dispatch latency the dominant speed factor. CuPy and Numba CUDA (low-overhead dispatcher) are faster than `nvmath-python` (heavy validation/descriptor layer).
- For **large matrices**, the actual computation takes hundreds of milliseconds. The dispatch latency is completely negligible, and throughput is solely determined by backend compute efficiency (cuBLAS achieving 14 TFLOPS vs. Numba achieving 2.4 TFLOPS).

---

## 7. Epilog Fusion: GEMM + Bias + ReLU

In deep learning, fully-connected (linear) layers are typically computed in three sequential steps:
1. **GEMM**: $y_{\text{raw}} = W \times x$
2. **Bias Addition**: $y_{\text{bias}} = y_{\text{raw}} + B$
3. **Activation Function**: $y = \text{ReLU}(y_{\text{bias}})$

### A. Performance Bottleneck of Unfused Execution
In a naive implementation, each step launches a separate GPU kernel. This requires writing intermediate results ($y_{\text{raw}}$ and $y_{\text{bias}}$) back to the GPU's High Bandwidth Memory (HBM) and reading them back for the next operation. This creates a severe memory bandwidth bottleneck.

### B. Epilog Fusion with nvmath-python
`nvmath-python` provides epilog configurations that fuse these operations into a single GPU kernel run via `cuBLASLt`:
- **`MatmulEpilog.RELU_BIAS`**: Performs GEMM, adds bias, and applies ReLU within the registers/SRAM of the GPU cores before writing the final result back to HBM.

### C. Benchmark Analysis (Tesla V100)
Using the configured dimensions of $100 \times 784$ weights and a batch size of $256$:
- **Naive Execution**: `2.576 ms`
- **Fused Execution**: `0.082 ms` ($\approx \mathbf{31.5\times}$ **speedup**)

Fusing these steps yields a massive $31\times$ performance increase on the forward pass by keeping intermediate results in the GPU registers/SRAM, completely bypassing redundant global memory read/write instructions.

---

## 8. Numba CUDA Version: `vector_add_numba.py`

This script demonstrates performing element-wise vector addition ($Z_i = X_i + Y_i$) on the GPU (Tesla V100) using **Numba**'s `@cuda.jit` compiler.

### Key Logic
1. **CUDA Kernel**:
   ```python
   @cuda.jit
   def vector_add_kernel(x, y, out):
       idx = cuda.grid(1)
       if idx < out.size:
           out[idx] = x[idx] + y[idx]
   ```
   - `@cuda.jit` compiles the Python function into a CUDA kernel.
   - `cuda.grid(1)` retrieves the 1D thread index corresponding to the current CUDA thread.
   - The boundary check `idx < out.size` prevents memory errors when vector sizes are not perfect multiples of the block size.
2. **Explicit Memory Allocation & Transfers**:
   - `cuda.to_device(x_cpu)` allocates GPU memory and copies the CPU vector to the device.
   - `cuda.device_array_like(out_cpu)` allocates uninitialized GPU memory of the same shape and type for the output vector.
3. **Execution Configuration**:
   - The grid size is dynamically calculated based on the vector length:
     ```python
     threads_per_block = 256
     blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
     ```
   - The kernel is launched asynchronously via `vector_add_kernel[blocks_per_grid, threads_per_block](...)`.
4. **Synchronization and Copy Back**:
   - `cuda.synchronize()` blocks the host CPU thread until the GPU finishes all operations.
   - `out_device.copy_to_host(out_cpu)` transfers the final results back to host memory for verification.

---

## 9. Numba CUDA Version: `matrix_mul_numba.py`

This script demonstrates performing matrix multiplication ($C = A \times B$) of size $1024 \times 1024$ on the GPU (Tesla V100) using **Numba**'s `@cuda.jit` compiler with a 2D grid/block layout.

### Key Logic
1. **2D CUDA Kernel**:
   ```python
   @cuda.jit
   def matmul_kernel(A, B, C):
       row, col = cuda.grid(2)
       if row < C.shape[0] and col < C.shape[1]:
           tmp = 0.0
           for k in range(A.shape[1]):
               tmp += A[row, k] * B[k, col]
           C[row, col] = tmp
   ```
   - `cuda.grid(2)` automatically calculates the 2D global index `(row, col)` for each thread using thread and block configurations.
   - The condition `row < C.shape[0] and col < C.shape[1]` ensures boundary protection for arbitrary matrix shapes.
   - The inner loop computes the dot product of row `row` of $A$ and column `col` of $B$.
2. **2D Thread/Grid Configuration**:
   - The execution is structured into blocks of size $(16, 16)$ threads.
   - The grid dimensions are computed dynamically:
     ```python
     threads_per_block = (16, 16)
     blocks_per_grid_x = (size + threads_per_block[0] - 1) // threads_per_block[0]
     blocks_per_grid_y = (size + threads_per_block[1] - 1) // threads_per_block[1]
     blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
     ```
3. **Execution, Warmup, and Benchmarking**:
   - Similar to the vector addition example, memory is transferred using `cuda.to_device(...)` and retrieved using `copy_to_host(...)`.
   - A warmup execution ensures compilation overhead is excluded from the benchmark timing measurements.

---

## 10. Numba CUDA Roofline Benchmark: `matrix_mul_numba_roofline.py`

This script profiles Numba's GPU matrix multiplication performance across sizes ranging from $32 \times 32$ to $16384 \times 16384$ on a Tesla V100 GPU.

### Shared-Memory Tiling Optimization
To scale efficiently to large matrix sizes without causing driver timeouts or massive latencies, `matrix_mul_numba_roofline.py` uses a **blocked shared-memory kernel**:
- **Cooperative Loading**: Threads in a block cooperatively load a $16 \times 16$ tile of $A$ and a $16 \times 16$ tile of $B$ into high-speed, on-chip **Shared Memory** (`cuda.shared.array`).
- **Synchronization**: `cuda.syncthreads()` acts as a barrier to ensure all threads in the block have finished loading the current tiles before compute starts, and another barrier ensures computing is done before loading the next tiles.
- **Arithmetic Intensity (AI)**:
  - Total FLOPs: $2 \times N^3$
  - Total Memory traffic: $3 \times N^2 \times 4$ bytes (read $A$, read $B$, write $C$).
  - As size $N$ increases, AI increases linearly ($\text{AI} = N/6$).
- **Performance Results**:
  - For small sizes, latency and compilation overhead keep GFLOPS low.
  - As matrix sizes grow, performance approaches the shared-memory tiling peak compute limit, leveling off at **~2.4 TFLOPS** (2,400 GFLOPS) on the Tesla V100 GPU.
  - This is compared directly in `roofline_comparison.png` against CuPy and stateful `nvmath-python` (which both use the highly-tuned, assembly-optimized `cuBLAS` library and reach **14 TFLOPS**).

---

## 11. cuTile Vector Addition Version: `vector_add_cutile.py`

This script demonstrates performing element-wise vector addition ($Z_i = X_i + Y_i$) on the GPU using NVIDIA's new **cuTile** Python DSL (`cuda.tile`) and block-level tile loading/storing.

### Hardware Context
Unlike the other benchmarks and Numba CUDA codes which were executed on a **Tesla V100 SXM2** GPU, the cuTile implementation was tested and run on an **NVIDIA H100** GPU using a local Python 3.13 environment supporting the newer `cuda.tile` namespace.

### Key Logic
1. **cuTile Kernel**:
   ```python
   @ct.kernel
   def vector_add_kernel(x, y, out):
       # Get the block ID (equivalent to blockIdx.x)
       block_id = ct.bid(0)
       
       # Load data tiles from global memory
       x_tile = ct.load(x, index=(block_id,), shape=(TILE_SIZE,))
       y_tile = ct.load(y, index=(block_id,), shape=(TILE_SIZE,))
       
       # Perform elementwise addition on the tiles
       result_tile = x_tile + y_tile
       
       # Store the result tile back to global memory
       ct.store(out, index=(block_id,), tile=result_tile)
   ```
   - `@ct.kernel` compiles the cuTile entry point.
   - `ct.bid(0)` gets the 1D block index.
   - **Block-level Tiles**: Instead of threads fetching individual elements element-by-element manually, cuTile operates directly on blocks of size `TILE_SIZE` (here `256`) using high-level vector/tile representations. `ct.load` and `ct.store` cooperatively load and store the whole tile from/to global memory.
2. **Grid Layout**:
   - `grid = (ct.cdiv(n, TILE_SIZE), 1, 1)` uses cuTile's ceiling division helper `ct.cdiv` to compute the number of blocks needed to process a vector of size `n` with tile size `TILE_SIZE`.
3. **Warmup & Launch**:
    - Launching the kernel uses `ct.launch` passing the stream, grid, kernel, and argument tuple.
    - The first launch is a warmup pass to ensure compilation completes before time measurement.

---

## 12. cuTile Matrix Multiplication Version: `matrix_mul_cutile.py`

This script demonstrates performing 2D matrix multiplication ($C = A \times B$) of size $1024 \times 1024$ on the GPU using NVIDIA's new **cuTile** Python DSL (`cuda.tile`).

### Hardware Context
Like the cuTile vector addition script, this was tested and run on an **NVIDIA H100 PCIe (80GB)** GPU, utilizing block-level tile loads, stores, and Tensor Core mma calculations.

### Key Logic
1. **2D Block Indices**:
   ```python
   bidx = ct.bid(0)
   bidy = ct.bid(1)
   ```
   Unlike standard thread-level indexing (`threadIdx`, `blockIdx`), cuTile works at block-tile granularity. `ct.bid(0)` and `ct.bid(1)` retrieve the 2D coordinates of the block.
2. **Accumulator Initialization**:
   ```python
   accumulator = ct.zeros(shape=(TILE_M, TILE_N), dtype=ct.float32)
   ```
   Initializes a tile accumulator of shape `(TILE_M, TILE_N)` (here `16x16`) with all zeros in register/SRAM.
3. **Tile Loop & MMA**:
   ```python
   for k in range(steps):
       a_tile = ct.load(A, index=(bidx, k), shape=(TILE_M, TILE_K))
       b_tile = ct.load(B, index=(k, bidy), shape=(TILE_K, TILE_N))
       accumulator = ct.mma(a_tile, b_tile, accumulator)
   ```
   - Loops step-by-step through the shared dimension $K$.
   - **`ct.load`**: Loads a 2D tile from memory. For $A$, the tile at block-row `bidx` and tile-column `k` of shape `(16, 16)` is loaded. For $B$, the tile at tile-row `k` and block-column `bidy` is loaded.
   - **`ct.mma`**: Performs the block-level matrix multiply-accumulate utilizing hardware Tensor Cores directly.
4. **`ct.store`**:
   ```python
   ct.store(C, index=(bidx, bidy), tile=accumulator)
   ```
   Writes the final accumulated result tile back to matrix $C$ at block coordinates `(bidx, bidy)` in global memory.

---

## 13. cuTile H100 Roofline Analysis: `matrix_mul_cutile_roofline.py`

This script profiles the performance of the cuTile matrix multiplication across sizes ranging from $32 \times 32$ to $16384 \times 16384$ and plots the measured performance against the theoretical bounds of the **NVIDIA H100 PCIe** GPU.

### A. Theoretical Roofline Bounds (H100 PCIe)
- **Peak FP32 Performance**: 51 TFLOPS (51,000 GFLOPS)
- **Peak Memory Bandwidth**: 2.0 TB/s (2,000 GB/s)
- **Ridge Point**:
  $$I_{\text{ridge}} = \frac{\text{Peak Compute}}{\text{Memory Bandwidth}} = \frac{51000 \text{ GFLOPS}}{2000 \text{ GB/s}} \approx 25.50 \text{ FLOP/Byte}$$
  If the Arithmetic Intensity (AI) is $\ge 25.50$ FLOP/Byte, execution is theoretically compute-bound. Since our sizes $N \ge 256$ have $\text{AI} \ge 42.67$ FLOP/Byte, the benchmarks fall strictly within the compute-bound region.

### B. Benchmark Results & Observations
The measured performance levels off at **~1015 GFLOPS** (1.015 TFLOPS) starting around size $4096 \times 4096$:

| Matrix Size | Arithmetic Intensity | Execution Time (s) | Throughput (GFLOPS) |
| :--- | :--- | :--- | :--- |
| **32x32** | 5.33 FLOP/B | 0.000025 s | 2.66 GFLOPS |
| **128x128** | 21.33 FLOP/B | 0.000019 s | 222.72 GFLOPS |
| **512x512** | 85.33 FLOP/B | 0.000315 s | 852.97 GFLOPS |
| **1024x1024** | 170.67 FLOP/B | 0.002356 s | 911.42 GFLOPS |
| **4096x4096** | 682.67 FLOP/B | 0.136846 s | 1,004.33 GFLOPS |
| **16384x16384** | 2730.67 FLOP/B | 8.667315 s | 1,014.86 GFLOPS |

### C. Performance Analysis: Naive cuTile vs. Peak Compute
While **1 TFLOPS** represents extremely fast execution (completing a $1024 \times 1024$ matmul in just $2.3$ milliseconds), it represents approximately $1.99\%$ of the H100 GPU's peak compute capacity ($51$ TFLOPS). This plateau is due to the simple nature of our naive kernel:
1. **No Shared Memory Pipelining**: In our loop, global memory tiles are loaded directly into the execution pipelines (`ct.load`). While the memory transfers happen, the Tensor Core computation pipeline stalls waiting for data. To bypass this, high-performance kernels use **double buffering** (or multi-stage pipelining) to prefetch the next tile's data into shared memory using asynchronous TMA (Tensor Memory Accelerator) instructions while the current tile is being multiplied.
2. **No Warp Specialization**: High-performance Hopper kernels dedicate separate sets of warps inside the SM for memory loading vs. math execution, running them concurrently.
3. **No L2 Cache Swizzling**: Thread blocks are dispatched to the GPU in linear raster order, which does not maximize cache reuse. Swizzling coordinates improves L2 cache hit rates significantly.

*Note: cuTile provides lower-level parameters to implement pipelining, swizzling, and warp specialization. Our example provides a clean, easy-to-read reference demonstrating correctness and baseline Tensor Core execution.*

