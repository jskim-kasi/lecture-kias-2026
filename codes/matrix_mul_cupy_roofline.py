import time
import csv
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

def benchmark_matrix_multiplication():
    # Sizes to benchmark: from 32 to 16384
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    results = []

    print("Starting matrix multiplication benchmarks...")
    for size in sizes:
        print(f"\nBenchmarking size {size}x{size}...")
        
        # Memory cleanup before allocation
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

        # Generate on CPU and transfer to GPU to bypass Curand issue
        try:
            A = cp.asarray(np.random.rand(size, size).astype(np.float32))
            B = cp.asarray(np.random.rand(size, size).astype(np.float32))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"Out of GPU Memory for size {size}x{size}!")
            break

        cp.cuda.Device(0).synchronize()

        # Warmup
        _ = A @ B
        cp.cuda.Device(0).synchronize()

        # Benchmark runs
        num_runs = 5
        timings = []
        for run in range(num_runs):
            start = time.perf_counter()
            _ = A @ B
            cp.cuda.Device(0).synchronize()
            timings.append(time.perf_counter() - start)

        avg_time = np.mean(timings)
        
        # Calculate performance metrics
        flops = 2 * (size ** 3)
        bytes_transferred = 3 * (size ** 2) * A.itemsize
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

    # Save results to CSV file
    with open("benchmark_cupy_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_time", "flops", "bytes", "ai", "gflops"])
        writer.writeheader()
        writer.writerows(results)

    return results

def plot_roofline(results):
    print("\nGenerating Roofline Model plot...")

    # Theoretical Limits of NVIDIA Tesla V100 SXM2 (32GB)
    peak_flops_gflops = 15700.0  # 15.7 TFLOPS
    memory_bandwidth_gbs = 900.0  # 900 GB/s (FLOP/Byte scaling factor)

    # Setup the plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Generate Arithmetic Intensity values for theoretical lines (from 1 to 10000 FLOP/Byte)
    ai_theoretical = np.logspace(0, 4, 1000)
    
    # Roofline boundary: min(Bandwidth * AI, Peak GFLOPS)
    roofline = np.minimum(memory_bandwidth_gbs * ai_theoretical, peak_flops_gflops)
    
    # Plot the Roofline limit
    plt.loglog(ai_theoretical, roofline, 'r-', linewidth=3, label='Theoretical Roofline Limit')
    
    # Plot the components separately for clarity
    plt.loglog(ai_theoretical, [peak_flops_gflops]*len(ai_theoretical), 'k--', alpha=0.5, label='Peak FP32 Performance (15.7 TFLOPS)')
    plt.loglog(ai_theoretical, memory_bandwidth_gbs * ai_theoretical, 'b--', alpha=0.5, label='Peak Memory Bandwidth (900 GB/s)')
    
    # Extract benchmark data points
    ai_measured = [r["ai"] for r in results]
    gflops_measured = [r["gflops"] for r in results]
    sizes = [r["size"] for r in results]
    
    # Plot measured benchmarks
    plt.scatter(ai_measured, gflops_measured, color='darkorange', marker='o', s=100, edgecolors='black', zorder=5, label='Measured Matrix Multiplication')
    
    # Formatting
    plt.title("Roofline Model - CuPy Matrix Multiplication on Tesla V100 GPU", fontsize=14, fontweight='bold')
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    plt.ylabel("Performance (GFLOPS)", fontsize=12)
    plt.grid(True, which="both", ls="-", color='lightgray')
    plt.xlim(1, 10000)
    plt.ylim(10, 30000)
    plt.legend(loc='lower right', fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("roofline_cupy.png")
    print("Roofline plot saved successfully as 'roofline_cupy.png'.")

def main():
    results = benchmark_matrix_multiplication()
    plot_roofline(results)

if __name__ == "__main__":
    main()
