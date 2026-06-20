import cuda.tile as ct
import cupy as cp
import numpy as np
import time

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

def main():
    # Matrix dimensions
    M, N, K = 1024, 1024, 1024
    
    print(f"Initializing matrices of size {M}x{N} and {N}x{K} on CPU...")
    # Initialize input matrices on CPU
    A_cpu = np.random.rand(M, K).astype(np.float32)
    B_cpu = np.random.rand(K, N).astype(np.float32)
    C_cpu = np.zeros((M, N)).astype(np.float32)
    
    # Allocate and copy data to GPU device memory
    print("Copying data to GPU device memory...")
    A_device = cp.asarray(A_cpu)
    B_device = cp.asarray(B_cpu)
    C_device = cp.zeros_like(A_device)
    
    # Determine grid size (number of tiles needed in 2D)
    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), 1)
    
    print(f"Launching cuTile matmul kernel with grid size {grid} and tile configuration ({TILE_M}x{TILE_N}x{TILE_K})...")
    
    # Warm-up to ensure JIT compilation is complete before timing
    ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel, (A_device, B_device, C_device))
    cp.cuda.Stream.null.synchronize()
    
    # Measure execution time
    start_time = time.perf_counter()
    ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel, (A_device, B_device, C_device))
    cp.cuda.Stream.null.synchronize()
    elapsed_time = time.perf_counter() - start_time
    
    print(f"Kernel execution took {elapsed_time * 1e3:.3f} ms.")
    
    # Calculate GFLOPS
    flops = 2 * (M * N * K)
    gflops = (flops / elapsed_time) * 1e-9
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    # Copy result back to CPU memory
    print("Copying result back to CPU memory...")
    C_cpu = cp.asnumpy(C_device)
    
    # Verify correctness against reference numpy multiplication
    print("Verifying correctness...")
    expected = A_cpu @ B_cpu
    match = np.allclose(C_cpu, expected, atol=1e-3, rtol=1e-3)
    print(f"Verification matching expected output: {match}")

if __name__ == "__main__":
    main()
