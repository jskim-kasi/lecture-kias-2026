import time
import csv
import gc
import numpy as np
import cupy as cp
import nvmath
from numba import cuda, float32
import matplotlib.pyplot as plt

# Block size for Numba shared memory tiling
TPB = 16

# Blocked CUDA kernel for matrix multiplication using shared memory
@cuda.jit
def matmul_blocked_kernel(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    row = cuda.blockIdx.y * cuda.blockDim.y + ty
    col = cuda.blockIdx.x * cuda.blockDim.x + tx

    tmp = 0.0

    # Loop over all tiles required to compute the C element
    for i in range((A.shape[1] + TPB - 1) // TPB):
        # Cooperatively load a tile of A into shared memory
        if row < A.shape[0] and (i * TPB + tx) < A.shape[1]:
            sA[ty, tx] = A[row, i * TPB + tx]
        else:
            sA[ty, tx] = 0.0

        # Cooperatively load a tile of B into shared memory
        if col < B.shape[1] and (i * TPB + ty) < B.shape[0]:
            sB[ty, tx] = B[i * TPB + ty, col]
        else:
            sB[ty, tx] = 0.0

        # Wait for all threads to load the tiles
        cuda.syncthreads()

        # Compute dot product contribution of this tile
        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]

        # Wait for all threads to finish computing before loading the next tile
        cuda.syncthreads()

    # Write the result to global memory
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

def run_cupy_benchmark(sizes):
    results = []
    print("\n--- Starting CuPy Matrix Multiplication Benchmarks ---")
    mempool = cp.get_default_memory_pool()
    for size in sizes:
        print(f"Benchmarking CuPy size {size}x{size}...")
        mempool.free_all_blocks()
        gc.collect()
        try:
            A = cp.asarray(np.random.rand(size, size).astype(np.float32))
            B = cp.asarray(np.random.rand(size, size).astype(np.float32))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"Out of GPU Memory for CuPy size {size}x{size}!")
            break
        cp.cuda.Device(0).synchronize()
        try:
            # Warmup
            _ = A @ B
            cp.cuda.Device(0).synchronize()
            # Benchmark runs
            num_runs = 5
            timings = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = A @ B
                cp.cuda.Device(0).synchronize()
                timings.append(time.perf_counter() - start)
            avg_time = np.mean(timings)
            flops = 2 * (size ** 3)
            bytes_transferred = 3 * (size ** 2) * 4
            ai = flops / bytes_transferred
            gflops = (flops / avg_time) * 1e-9
            results.append({
                "size": size,
                "avg_time": avg_time,
                "flops": flops,
                "bytes": bytes_transferred,
                "ai": ai,
                "gflops": gflops
            })
        except Exception as e:
            print(f"Error benchmarking CuPy size {size}x{size}: {e}")
            break
        finally:
            if 'A' in locals(): del A
            if 'B' in locals(): del B
            mempool.free_all_blocks()
            gc.collect()
            
    csv_file = "benchmark_cupy_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_time", "flops", "bytes", "ai", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CuPy benchmark results to {csv_file}")
    return results

def run_nvmath_benchmark(sizes):
    results = []
    print("\n--- Starting nvmath Matrix Multiplication Benchmarks ---")
    mempool = cp.get_default_memory_pool()
    for size in sizes:
        print(f"Benchmarking nvmath size {size}x{size}...")
        mempool.free_all_blocks()
        gc.collect()
        try:
            A = cp.asarray(np.random.rand(size, size).astype(np.float32))
            B = cp.asarray(np.random.rand(size, size).astype(np.float32))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"Out of GPU Memory for nvmath size {size}x{size}!")
            break
        cp.cuda.Device(0).synchronize()
        try:
            with nvmath.linalg.advanced.Matmul(A, B) as mm:
                mm.plan()
                # Warmup
                _ = mm.execute()
                cp.cuda.Device(0).synchronize()
                # Benchmark runs
                num_runs = 5
                timings = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = mm.execute()
                    cp.cuda.Device(0).synchronize()
                    timings.append(time.perf_counter() - start)
                avg_time = np.mean(timings)
                flops = 2 * (size ** 3)
                bytes_transferred = 3 * (size ** 2) * 4
                ai = flops / bytes_transferred
                gflops = (flops / avg_time) * 1e-9
                results.append({
                    "size": size,
                    "avg_time": avg_time,
                    "flops": flops,
                    "bytes": bytes_transferred,
                    "ai": ai,
                    "gflops": gflops
                })
        except Exception as e:
            print(f"Error benchmarking nvmath size {size}x{size}: {e}")
            break
        finally:
            if 'A' in locals(): del A
            if 'B' in locals(): del B
            mempool.free_all_blocks()
            gc.collect()
            
    csv_file = "benchmark_nvmath_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_time", "flops", "bytes", "ai", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved nvmath benchmark results to {csv_file}")
    return results

def run_numba_benchmark(sizes):
    results = []
    print("\n--- Starting Numba CUDA Matrix Multiplication Benchmarks ---")
    for size in sizes:
        print(f"Benchmarking Numba size {size}x{size}...")
        cuda.current_context().deallocations.clear()
        gc.collect()
        try:
            A_cpu = np.random.rand(size, size).astype(np.float32)
            B_cpu = np.random.rand(size, size).astype(np.float32)
            C_cpu = np.zeros((size, size)).astype(np.float32)
            A_device = cuda.to_device(A_cpu)
            B_device = cuda.to_device(B_cpu)
            C_device = cuda.device_array_like(C_cpu)
        except Exception as e:
            print(f"Memory allocation error for Numba size {size}x{size}: {e}")
            break
        threads_per_block = (TPB, TPB)
        blocks_per_grid_x = (size + TPB - 1) // TPB
        blocks_per_grid_y = (size + TPB - 1) // TPB
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        try:
            # Warmup
            matmul_blocked_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
            cuda.synchronize()
            # Benchmark runs
            num_runs = 5
            timings = []
            for _ in range(num_runs):
                start = time.perf_counter()
                matmul_blocked_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
                cuda.synchronize()
                timings.append(time.perf_counter() - start)
            avg_time = np.mean(timings)
            flops = 2 * (size ** 3)
            bytes_transferred = 3 * (size ** 2) * 4
            ai = flops / bytes_transferred
            gflops = (flops / avg_time) * 1e-9
            results.append({
                "size": size,
                "avg_time": avg_time,
                "flops": flops,
                "bytes": bytes_transferred,
                "ai": ai,
                "gflops": gflops
            })
        except Exception as e:
            print(f"Error benchmarking Numba size {size}x{size}: {e}")
            break
        finally:
            if 'A_device' in locals(): del A_device
            if 'B_device' in locals(): del B_device
            if 'C_device' in locals(): del C_device
            if 'A_cpu' in locals(): del A_cpu
            if 'B_cpu' in locals(): del B_cpu
            if 'C_cpu' in locals(): del C_cpu
            cuda.current_context().deallocations.clear()
            gc.collect()
            
    csv_file = "benchmark_numba_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_time", "flops", "bytes", "ai", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved Numba benchmark results to {csv_file}")
    return results

def load_results(csv_path):
    ai = []
    gflops = []
    sizes = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ai.append(float(row['ai']))
            gflops.append(float(row['gflops']))
            sizes.append(int(row['size']))
    return ai, gflops, sizes

def plot_comparison():
    cupy_csv = "benchmark_cupy_results.csv"
    nvmath_csv = "benchmark_nvmath_results.csv"
    numba_csv = "benchmark_numba_results.csv"
    
    print("\n--- Loading Results for Plotting ---")
    cupy_ai, cupy_gflops, cupy_sizes = load_results(cupy_csv)
    nvmath_ai, nvmath_gflops, nvmath_sizes = load_results(nvmath_csv)
    numba_ai, numba_gflops, numba_sizes = load_results(numba_csv)
    
    # Theoretical Limits of NVIDIA Tesla V100 SXM2 (32GB)
    peak_flops_gflops = 15700.0  # 15.7 TFLOPS
    memory_bandwidth_gbs = 900.0  # 900 GB/s (FLOP/Byte scaling factor)

    # Setup the plot
    plt.figure(figsize=(11, 7), dpi=150)
    
    # Generate Arithmetic Intensity values for theoretical lines (from 1 to 10000 FLOP/Byte)
    ai_theoretical = np.logspace(0, 4, 1000)
    
    # Roofline boundary
    roofline = np.minimum(memory_bandwidth_gbs * ai_theoretical, peak_flops_gflops)
    
    # Plot the Roofline limit
    plt.loglog(ai_theoretical, roofline, 'r-', linewidth=3, label='Theoretical Roofline Limit')
    
    # Plot the components separately
    plt.loglog(ai_theoretical, [peak_flops_gflops]*len(ai_theoretical), 'k--', alpha=0.5, label='Peak FP32 Performance (15.7 TFLOPS)')
    plt.loglog(ai_theoretical, memory_bandwidth_gbs * ai_theoretical, 'b--', alpha=0.5, label='Peak Memory Bandwidth (900 GB/s)')
    
    # Plot CuPy measured benchmarks
    plt.scatter(cupy_ai, cupy_gflops, color='darkorange', marker='o', s=120, edgecolors='black', zorder=5, label='CuPy Matrix Multiplication')
    
    # Plot nvmath measured benchmarks
    plt.scatter(nvmath_ai, nvmath_gflops, color='teal', marker='^', s=120, edgecolors='black', zorder=6, label='nvmath-python (Stateful)')
    
    # Plot Numba measured benchmarks
    plt.scatter(numba_ai, numba_gflops, color='purple', marker='s', s=120, edgecolors='black', zorder=7, label='Numba (Shared-Memory Tile)')
    
    # Formatting
    plt.title("Roofline Model Comparison: CuPy vs nvmath vs Numba (Tesla V100 GPU)", fontsize=14, fontweight='bold')
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    plt.ylabel("Performance (GFLOPS)", fontsize=12)
    plt.grid(True, which="both", ls="-", color='lightgray')
    plt.xlim(1, 10000)
    plt.ylim(10, 30000)
    plt.legend(loc='lower right', fontsize=11)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("roofline_comparison.png")
    print("Roofline comparison plot saved successfully as 'roofline_comparison.png'.")

def main():
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    # Run the three benchmarks
    run_cupy_benchmark(sizes)
    run_nvmath_benchmark(sizes)
    run_numba_benchmark(sizes)
    
    # Generate unified comparison plot
    plot_comparison()

if __name__ == "__main__":
    main()
