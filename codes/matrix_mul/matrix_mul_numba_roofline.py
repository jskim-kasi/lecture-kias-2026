import time
import csv
import gc
import numpy as np
from numba import cuda, float32
import matplotlib.pyplot as plt

# Block size for shared memory tiling
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

def benchmark_matrix_multiplication():
    # Sizes to benchmark: from 32 to 16384 in units of power of 2
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    print("Starting Numba CUDA matrix multiplication benchmarks...")
    for size in sizes:
        print(f"\nBenchmarking size {size}x{size}...")
        
        # Memory cleanup before allocation
        cuda.current_context().deallocations.clear()
        gc.collect()

        try:
            # Generate random arrays on CPU and copy to GPU
            A_cpu = np.random.rand(size, size).astype(np.float32)
            B_cpu = np.random.rand(size, size).astype(np.float32)
            C_cpu = np.zeros((size, size)).astype(np.float32)

            A_device = cuda.to_device(A_cpu)
            B_device = cuda.to_device(B_cpu)
            C_device = cuda.device_array_like(C_cpu)
        except Exception as e:
            print(f"Memory allocation error for size {size}x{size}: {e}")
            break

        # Thread blocks configuration
        threads_per_block = (TPB, TPB)
        blocks_per_grid_x = (size + TPB - 1) // TPB
        blocks_per_grid_y = (size + TPB - 1) // TPB
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        try:
            # Warmup phase
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
            
            # Calculate performance metrics
            flops = 2 * (size ** 3)
            bytes_transferred = 3 * (size ** 2) * 4 # float32 has 4 bytes
            ai = flops / bytes_transferred
            gflops = (flops / avg_time) * 1e-9

            print(f"Size {size}x{size} completed. Avg Time: {avg_time:.6f} s, Performance: {gflops:.2f} GFLOPs, AI: {ai:.2f} FLOP/Byte")

            results.append({
                "size": size,
                "avg_time": avg_time,
                "flops": flops,
                "bytes": bytes_transferred,
                "ai": ai,
                "gflops": gflops
            })
        except Exception as e:
            print(f"Error benchmarking size {size}x{size}: {e}")
            break
        finally:
            # Cleanup references
            if 'A_device' in locals(): del A_device
            if 'B_device' in locals(): del B_device
            if 'C_device' in locals(): del C_device
            if 'A_cpu' in locals(): del A_cpu
            if 'B_cpu' in locals(): del B_cpu
            if 'C_cpu' in locals(): del C_cpu
            cuda.current_context().deallocations.clear()
            gc.collect()

    # Save results to CSV file
    csv_file = "benchmark_numba_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_time", "flops", "bytes", "ai", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved benchmark results to {csv_file}")

    return results

def plot_roofline(results):
    print("\nGenerating Roofline Model plot...")

    # Theoretical Limits of NVIDIA Tesla V100 SXM2 (32GB)
    peak_flops_gflops = 15700.0  # 15.7 TFLOPS
    memory_bandwidth_gbs = 900.0  # 900 GB/s

    # Setup the plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Generate Arithmetic Intensity values for theoretical lines (from 1 to 10000 FLOP/Byte)
    ai_theoretical = np.logspace(0, 4, 1000)
    
    # Roofline boundary
    roofline = np.minimum(memory_bandwidth_gbs * ai_theoretical, peak_flops_gflops)
    
    # Plot the Roofline limit
    plt.loglog(ai_theoretical, roofline, 'r-', linewidth=3, label='Theoretical Roofline Limit')
    
    # Plot the components separately for clarity
    plt.loglog(ai_theoretical, [peak_flops_gflops]*len(ai_theoretical), 'k--', alpha=0.5, label='Peak FP32 Performance (15.7 TFLOPS)')
    plt.loglog(ai_theoretical, memory_bandwidth_gbs * ai_theoretical, 'b--', alpha=0.5, label='Peak Memory Bandwidth (900 GB/s)')
    
    # Extract benchmark data points
    ai_measured = [r["ai"] for r in results]
    gflops_measured = [r["gflops"] for r in results]
    
    # Plot measured benchmarks
    plt.scatter(ai_measured, gflops_measured, color='purple', marker='s', s=100, edgecolors='black', zorder=5, label='Numba Shared-Memory Matmul')
    
    # Formatting
    plt.title("Roofline Model - Numba CUDA Matrix Multiplication on Tesla V100 GPU", fontsize=14, fontweight='bold')
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    plt.ylabel("Performance (GFLOPS)", fontsize=12)
    plt.grid(True, which="both", ls="-", color='lightgray')
    plt.xlim(1, 10000)
    plt.ylim(10, 30000)
    plt.legend(loc='lower right', fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("roofline_numba.png")
    print("Roofline plot saved successfully as 'roofline_numba.png'.")

def main():
    results = benchmark_matrix_multiplication()
    if results:
        plot_roofline(results)

if __name__ == "__main__":
    main()
