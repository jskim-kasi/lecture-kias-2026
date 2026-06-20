import time
import numpy as np
import cupy as cp
import nvmath

def main():
    # Set the size of the matrices
    size = 1024
    print(f"Initializing two {size}x{size} matrices on the GPU (using CuPy & nvmath)...")

    # Allocate random float32 matrices (numpy-to-cupy copy to avoid Curand issue)
    A = cp.asarray(np.random.rand(size, size).astype(np.float32))
    B = cp.asarray(np.random.rand(size, size).astype(np.float32))

    # Synchronize device
    cp.cuda.Device(0).synchronize()

    # Stateful Matmul context
    with nvmath.linalg.advanced.Matmul(A, B) as mm:
        # Planning phase
        print("Planning GEMM execution...")
        mm.plan()
        
        # Warmup phase
        print("Warming up nvmath-python GEMM...")
        _ = mm.execute()
        cp.cuda.Device(0).synchronize()

        print("Performing matrix multiplication A @ B using stateful nvmath-python...")

        # Benchmark runs
        num_runs = 5
        timings = []
        for run in range(num_runs):
            start = time.perf_counter()
            C = mm.execute()
            cp.cuda.Device(0).synchronize()
            timings.append(time.perf_counter() - start)

        avg_time = np.mean(timings)

        # Calculate GFLOPs performance
        flops = 2 * (size ** 3)
        gflops = (flops / avg_time) * 1e-9

        # Print results
        print("\n---------------- Results ----------------")
        print(f"Result shape: {C.shape}")
        print(f"Result dtype: {C.dtype}")
        print(f"Avg Processing time: {avg_time:.6f} seconds")
        print(f"Performance: {gflops:.2f} GFLOPs")
        print(f"Total FLOPs: {flops:,} FLOPs")
        print("-----------------------------------------")

if __name__ == "__main__":
    main()
