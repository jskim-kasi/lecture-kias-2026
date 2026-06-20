import time
import cupy as cp

def main():
    # Set the size of the matrices
    size = 1024
    print(f"Initializing two {size}x{size} matrices on the GPU (CuPy)...")

    # Allocate random float32 matrices
    try:
        # Attempt direct GPU random allocation
        A = cp.random.rand(size, size).astype(cp.float32)
        B = cp.random.rand(size, size).astype(cp.float32)
    except Exception as e:
        print(f"Direct GPU generation failed ({e}). Falling back to CPU host generation...")
        import numpy as np
        A = cp.asarray(np.random.rand(size, size).astype(np.float32))
        B = cp.asarray(np.random.rand(size, size).astype(np.float32))

    # Ensure memory allocation and initialization are completed
    cp.cuda.Device(0).synchronize()

    # Warmup to compile/initialize GPU libraries (like cuBLAS)
    print("Warming up GPU library...")
    _ = A @ B
    cp.cuda.Device(0).synchronize()

    print("Performing matrix multiplication A @ B on GPU...")

    # Measure start time
    start_time = time.perf_counter()

    # Perform matrix multiplication on GPU
    C = A @ B

    # Synchronize to make sure GPU is finished before stopping the timer
    cp.cuda.Device(0).synchronize()

    # Measure end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Calculate Arithmetic Intensity (AI)
    # FLOPs = 2 * N^3 (for standard matrix multiplication)
    # Bytes Transferred = (Read A + Read B + Write C) * sizeof(float32) = 3 * N^2 * 4
    flops = 2 * (size ** 3)
    bytes_transferred = 3 * (size ** 2) * A.itemsize
    arithmetic_intensity = flops / bytes_transferred

    # Calculate GFLOPs performance
    gflops = (flops / elapsed_time) * 1e-9

    # Print results
    print("\n---------------- Results ----------------")
    print(f"Result shape: {C.shape}")
    print(f"Result dtype: {C.dtype}")
    print(f"Processing time: {elapsed_time:.6f} seconds")
    print(f"Performance: {gflops:.2f} GFLOPs")
    print(f"Total FLOPs: {flops:,} FLOPs")
    print(f"Total Memory Traffic: {bytes_transferred:,} Bytes ({bytes_transferred / (1024**2):.2f} MB)")
    print(f"Arithmetic Intensity (AI): {arithmetic_intensity:.2f} FLOP/Byte")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
