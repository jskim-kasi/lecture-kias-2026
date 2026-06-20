import time
import csv
import gc
import os
import numpy as np
import cupy as cp
import cuda.tile as ct
import matplotlib.pyplot as plt

# Tile dimensions for matrix multiplication block
TILE_M = 16
TILE_N = 16
TILE_K = 16

# CUDA kernel for matrix multiplication using cuTile
@ct.kernel
def matmul_kernel(A, B, C):
    # Get the 2D block index
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    
    # Initialize the accumulator tile to zero
    accumulator = ct.zeros(shape=(TILE_M, TILE_N), dtype=ct.float32)
    
    # Calculate the number of steps along the K dimension
    K = A.shape[1]
    steps = ct.cdiv(K, TILE_K)
    
    # Loop over the tiles of K
    for k in range(steps):
        # Load tiles from A and B from global memory
        a_tile = ct.load(A, index=(bidx, k), shape=(TILE_M, TILE_K))
        b_tile = ct.load(B, index=(k, bidy), shape=(TILE_K, TILE_N))
        
        # Matrix multiply-accumulate using Tensor Cores
        accumulator = ct.mma(a_tile, b_tile, accumulator)
        
    # Store the accumulated tile back to global memory C
    ct.store(C, index=(bidx, bidy), tile=accumulator)

def benchmark_matrix_multiplication():
    # Sizes to benchmark: from 32 to 16384 (all multiples of 16)
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    print("Starting cuTile matrix multiplication benchmarks...")
    for size in sizes:
        print(f"\nBenchmarking size {size}x{size}...")
        
        # Memory cleanup before allocation
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        gc.collect()

        # Generate on CPU and transfer to GPU to bypass Curand issue
        try:
            A_cpu = np.random.rand(size, size).astype(np.float32)
            B_cpu = np.random.rand(size, size).astype(np.float32)
            A = cp.asarray(A_cpu)
            B = cp.asarray(B_cpu)
            C = cp.zeros_like(A)
            del A_cpu, B_cpu
        except cp.cuda.memory.OutOfMemoryError:
            print(f"Out of GPU Memory for size {size}x{size}!")
            break

        cp.cuda.Device(0).synchronize()

        # Grid configuration
        grid = (ct.cdiv(size, TILE_M), ct.cdiv(size, TILE_N), 1)

        # Warmup
        ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel, (A, B, C))
        cp.cuda.Stream.null.synchronize()

        # Benchmark runs
        num_runs = 5
        timings = []
        for _ in range(num_runs):
            start = time.perf_counter()
            ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel, (A, B, C))
            cp.cuda.Stream.null.synchronize()
            timings.append(time.perf_counter() - start)

        avg_time = np.mean(timings)
        
        # Calculate performance metrics
        flops = 2 * (size ** 3)
        bytes_transferred = 3 * (size ** 2) * 4 # 4 bytes per float32
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
        
        # Free memory references
        del A, B, C
        mempool.free_all_blocks()
        gc.collect()

    # Save results to CSV file relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "benchmark_cutile_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_time", "flops", "bytes", "ai", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved cuTile benchmark results to {csv_file}")

    return results, script_dir

def plot_roofline(results, script_dir):
    print("\nGenerating Roofline Model plot...")

    # Theoretical Limits of NVIDIA H100 PCIe (80GB)
    peak_flops_gflops = 51000.0  # 51 TFLOPS
    memory_bandwidth_gbs = 2000.0  # 2000 GB/s (FLOP/Byte scaling factor)

    # Setup the plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Generate Arithmetic Intensity values for theoretical lines (from 1 to 10000 FLOP/Byte)
    ai_theoretical = np.logspace(0, 4, 1000)
    
    # Roofline boundary: min(Bandwidth * AI, Peak GFLOPS)
    roofline = np.minimum(memory_bandwidth_gbs * ai_theoretical, peak_flops_gflops)
    
    # Plot the Roofline limit
    plt.loglog(ai_theoretical, roofline, 'r-', linewidth=3, label='Theoretical Roofline Limit')
    
    # Plot the components separately for clarity
    plt.loglog(ai_theoretical, [peak_flops_gflops]*len(ai_theoretical), 'k--', alpha=0.5, label='Peak FP32 Performance (51 TFLOPS)')
    plt.loglog(ai_theoretical, memory_bandwidth_gbs * ai_theoretical, 'b--', alpha=0.5, label='Peak Memory Bandwidth (2.0 TB/s)')
    
    # Extract benchmark data points
    ai_measured = [r["ai"] for r in results]
    gflops_measured = [r["gflops"] for r in results]
    
    # Plot measured benchmarks
    plt.scatter(ai_measured, gflops_measured, color='teal', marker='^', s=100, edgecolors='black', zorder=5, label='Measured cuTile matmul')
    
    # Formatting
    plt.title("Roofline Model - cuTile Matrix Multiplication on H100 GPU", fontsize=14, fontweight='bold')
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    plt.ylabel("Performance (GFLOPS)", fontsize=12)
    plt.grid(True, which="both", ls="-", color='lightgray')
    plt.xlim(1, 10000)
    plt.ylim(10, 80000)
    plt.legend(loc='lower right', fontsize=10)
    
    # Save the plot relative to script location
    plot_file = os.path.join(script_dir, "roofline_cutile.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Roofline plot saved successfully as '{plot_file}'.")

def main():
    results, script_dir = benchmark_matrix_multiplication()
    plot_roofline(results, script_dir)

if __name__ == "__main__":
    main()
