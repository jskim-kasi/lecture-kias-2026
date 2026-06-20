import time
import numpy as np
import cupy
import nvmath
from nvmath.linalg.advanced import Matmul, MatmulEpilog

def main():
    num_inputs, num_outputs = 784, 100
    batch_size = 256

    print(f"Initializing arrays (Outputs: {num_outputs}, Inputs: {num_inputs}, Batch: {batch_size})...")

    # Directly initialize in GPU memory
    weights = cupy.random.rand(num_outputs, num_inputs).astype(cupy.float32)
    bias = cupy.random.rand(num_outputs).astype(cupy.float32)
    x = cupy.zeros((num_inputs, batch_size), dtype=cupy.float32)

    print("\nSetting up Matmul objects and plans...")
    
    # Naive path (GEMM only, planned)
    mm_naive = Matmul(weights, x)
    mm_naive.plan()
    
    # Warmup naive path
    _ = mm_naive.execute()
    cupy.cuda.Device(0).synchronize()

    # Fused path (GEMM + Bias + ReLU, planned)
    mm_relu_bias = Matmul(weights, x)
    mm_relu_bias.plan(epilog=MatmulEpilog.RELU_BIAS, epilog_inputs={"bias": bias})
    
    # Warmup fused path
    _ = mm_relu_bias.execute()
    cupy.cuda.Device(0).synchronize()

    # --- Benchmarking ---
    num_runs = 100
    print(f"\n--- Benchmarking ({num_runs} runs) ---")

    # Benchmarking Naive (Unfused forward pass)
    naive_timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        y = mm_naive.execute()
        y += bias[:, cupy.newaxis]
        y[y < 0] = 0
        cupy.cuda.Device(0).synchronize()
        naive_timings.append(time.perf_counter() - start)
    avg_naive = np.mean(naive_timings) * 1000  # ms
    print(f"Naive (Unfused) Fwd: {avg_naive:.3f} ms")

    # Benchmarking Fused (RELU_BIAS execution)
    relu_bias_timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = mm_relu_bias.execute()
        cupy.cuda.Device(0).synchronize()
        relu_bias_timings.append(time.perf_counter() - start)
    avg_relu_bias = np.mean(relu_bias_timings) * 1000  # ms
    print(f"Fused RELU_BIAS:     {avg_relu_bias:.3f} ms (Speedup: {avg_naive / avg_relu_bias:.2f}x)")

if __name__ == "__main__":
    main()
