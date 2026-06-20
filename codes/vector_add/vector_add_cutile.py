import cuda.tile as ct
import cupy as cp
import numpy as np
import time

TILE_SIZE = 256

# CUDA kernel for vector addition using cuTile
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

def main():
    # Array size (must be a multiple of TILE_SIZE for clean tile loading)
    n = 1024000
    
    print(f"Initializing vectors of size {n} on CPU...")
    # Initialize input vectors on CPU
    x_cpu = np.random.rand(n).astype(np.float32)
    y_cpu = np.random.rand(n).astype(np.float32)
    out_cpu = np.zeros(n).astype(np.float32)
    
    # Allocate and copy data to GPU device memory
    print("Copying data to GPU device memory...")
    x_device = cp.asarray(x_cpu)
    y_device = cp.asarray(y_cpu)
    out_device = cp.zeros_like(x_device)
    
    # Determine grid size (number of tiles needed)
    grid = (ct.cdiv(n, TILE_SIZE), 1, 1)
    
    print(f"Launching cuTile kernel with grid size {grid} and tile size {TILE_SIZE}...")
    
    # Warm-up to ensure compiler compilation is complete before timing
    ct.launch(cp.cuda.get_current_stream(), grid, vector_add_kernel, (x_device, y_device, out_device))
    cp.cuda.Stream.null.synchronize()
    
    # Measure execution time
    start_time = time.perf_counter()
    ct.launch(cp.cuda.get_current_stream(), grid, vector_add_kernel, (x_device, y_device, out_device))
    cp.cuda.Stream.null.synchronize()
    elapsed_time = time.perf_counter() - start_time
    
    print(f"Kernel execution took {elapsed_time * 1e3:.3f} ms.")
    
    # Copy result back to CPU memory
    print("Copying result back to CPU memory...")
    out_cpu = cp.asnumpy(out_device)
    
    # Verify correctness
    expected = x_cpu + y_cpu
    match = np.allclose(out_cpu, expected, atol=1e-6)
    print(f"Verification matching expected output: {match}")

if __name__ == "__main__":
    main()
