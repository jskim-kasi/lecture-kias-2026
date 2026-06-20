import numpy as np
from numba import cuda

# CUDA kernel for vector addition
@cuda.jit
def vector_add_kernel(x, y, out):
    # Get the global position of the current thread
    idx = cuda.grid(1)
    
    # Check if the thread index is within array bounds
    if idx < out.size:
        out[idx] = x[idx] + y[idx]

def main():
    # Array size
    n = 1000000
    
    print(f"Initializing vectors of size {n} on CPU...")
    # Initialize input vectors on CPU
    x_cpu = np.random.rand(n).astype(np.float32)
    y_cpu = np.random.rand(n).astype(np.float32)
    out_cpu = np.zeros(n).astype(np.float32)
    
    # Allocate and copy data to GPU device memory
    print("Copying data to GPU device memory...")
    x_device = cuda.to_device(x_cpu)
    y_device = cuda.to_device(y_cpu)
    out_device = cuda.device_array_like(out_cpu)
    
    # Set execution configuration (blocks and threads)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    print(f"Launching kernel with {blocks_per_grid} blocks of {threads_per_block} threads...")
    # Launch CUDA kernel
    vector_add_kernel[blocks_per_grid, threads_per_block](x_device, y_device, out_device)
    
    # Wait for the GPU to finish execution
    cuda.synchronize()
    
    # Copy result back to CPU memory
    print("Copying result back to CPU memory...")
    out_device.copy_to_host(out_cpu)
    
    # Verify correctness
    expected = x_cpu + y_cpu
    match = np.allclose(out_cpu, expected, atol=1e-6)
    print(f"Verification matching expected output: {match}")

if __name__ == "__main__":
    main()
