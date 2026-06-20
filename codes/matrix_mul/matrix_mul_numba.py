import numpy as np
from numba import cuda
import time

# Naive 2D CUDA kernel for matrix multiplication
@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def main():
    size = 1024
    print(f"Initializing two {size}x{size} matrices on CPU...")
    # Initialize matrices on CPU
    A_cpu = np.random.rand(size, size).astype(np.float32)
    B_cpu = np.random.rand(size, size).astype(np.float32)
    C_cpu = np.zeros((size, size)).astype(np.float32)
    
    # Allocate and copy data to GPU device memory
    print("Copying data to GPU device memory...")
    A_device = cuda.to_device(A_cpu)
    B_device = cuda.to_device(B_cpu)
    C_device = cuda.device_array_like(C_cpu)
    
    # Execution configuration (2D grid of 2D blocks)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (size + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (size + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Warmup kernel
    print("Warming up kernel...")
    matmul_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    
    # Benchmark runs
    num_runs = 10
    print(f"Performing matrix multiplication A @ B using Numba CUDA ({num_runs} runs)...")
    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        matmul_kernel[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
        cuda.synchronize()
        timings.append(time.perf_counter() - start)
    
    avg_time = np.mean(timings)
    print(f"Avg Processing time: {avg_time:.6f} seconds")
    
    # Copy result back to CPU memory
    print("Copying result back to CPU memory...")
    C_device.copy_to_host(C_cpu)
    
    # Verify correctness
    expected = A_cpu @ B_cpu
    match = np.allclose(C_cpu, expected, atol=1e-4)
    print(f"Verification matching expected output: {match}")

if __name__ == "__main__":
    main()
