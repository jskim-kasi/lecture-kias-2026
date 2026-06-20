import time
import numpy as np

def main():
    # Set the size of the matrices
    size = 1024
    print(f"Initializing two {size}x{size} matrices with random values...")

    # Generate two random 1024x1024 float32 matrices
    # numpy.random.rand returns float64 by default, so we cast to float32
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    print("Performing matrix multiplication A @ B...")

    # Measure start time
    start_time = time.perf_counter()

    # Perform matrix multiplication
    C = A @ B

    # Measure end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print results
    print(f"Matrix multiplication completed successfully.")
    print(f"Result shape: {C.shape}")
    print(f"Result dtype: {C.dtype}")
    print(f"Time taken: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
