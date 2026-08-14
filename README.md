# High Performance CUDA Kernels for Transformer Workloads

Implementation and optimization of core GPU kernels used in Transformer workloads:
**GEMM, Multi-Layer Perceptron (MLP), and FlashAttention**.

The project studies how GPU memory hierarchy, data reuse, and parallel execution
affect kernel performance by progressively optimizing CUDA implementations and
benchmarking them against optimized libraries such as **cuBLAS, PyTorch, and
FlashAttention**.

## Project Overview

The project consists of three major components:

1. **High-Performance CUDA Matrix Multiplication (GEMM)**
2. **CUDA Multi-Layer Perceptron (MLP)**
3. **CUDA FlashAttention**

The implementations are developed from scratch in CUDA C++ to understand the
optimization principles used by high-performance GPU libraries. The project
progressively optimizes kernels instead of relying directly on vendor
implementations.

## Key Optimizations

The GEMM implementation is optimized through the following stages:

- **Memory coalescing** for efficient global-memory accesses.
- **Shared-memory tiling** to increase data reuse and arithmetic intensity.
- **Register blocking** to reuse data and increase instruction-level parallelism.
- **Warp-level tiling** for improved workload distribution and GPU utilization.

These optimizations progressively reduce global-memory traffic and improve
computational throughput.

## CUDA GEMM

The project begins with a naive matrix multiplication kernel in which each
CUDA thread computes one output element.

The optimization pipeline is:

```text
Naive GEMM
    |
    v
Memory Coalescing
    |
    v
Shared-Memory Tiling
    |
    v
Register Blocking
    |
    v
Warp Tiling
```

The final warp-tiled implementation achieves **1.35 TFLOPS** and approximately
**15.93x speedup** over the naive implementation.

| Kernel | TFLOPS | Speedup |
|---|---:|---:|
| Naive | 0.085 | 1.00x |
| Memory Coalescing | 0.405 | 4.77x |
| Shared Memory Tiling | 0.377 | 4.45x |
| Register Blocking | 0.466 | 5.50x |
| Warp Tiling | 1.350 | 15.93x |
| cuBLAS | 3.558 | 41.99x |

The final implementation reaches approximately **37.9% of cuBLAS throughput**.

## CUDA Multi-Layer Perceptron

The MLP is implemented entirely using custom CUDA kernels and builds on the
optimized GEMM implementation.

It supports:

- Fully connected linear layers
- ReLU activation
- Forward propagation
- Backward propagation
- Mean Squared Error (MSE) loss
- Stochastic Gradient Descent (SGD) updates
- Custom gradient and parameter-update kernels

The forward pass uses the optimized GEMM kernel for its linear layers, while
activation and bias operations execute on the GPU.

The backward pass includes custom CUDA kernels for:

- Matrix transposition
- Gradient computation
- ReLU backpropagation
- Bias-gradient accumulation
- MSE loss computation
- SGD parameter updates

Keeping intermediate tensors on the GPU avoids unnecessary host-device
transfers during training.

## FlashAttention

The project also implements **FlashAttention** using CUDA shared-memory
tiling and online softmax.

Conventional attention explicitly materializes the complete attention matrix,
whose memory requirement grows as **O(N²)** with sequence length.

The implementation instead:

1. Divides query, key, and value matrices into tiles.
2. Loads tiles cooperatively into shared memory.
3. Computes attention scores block-by-block.
4. Maintains running softmax statistics using online normalization.
5. Accumulates output without materializing the complete attention matrix.

This reduces the auxiliary memory requirement to **O(N)** while preserving the
exact attention computation.

## Experimental Evaluation

Experiments were conducted using:

- **GPU:** NVIDIA Tesla T4, 16 GB GDDR6
- **CUDA:** CUDA 12.08
- **Compiler:** `nvcc` with `-O3`
- **Libraries:** CUDA Runtime API, cuBLAS, PyTorch

The GEMM, MLP, and FlashAttention kernels were benchmarked independently.
Multiple warm-up iterations were performed before measurement, and execution
times were averaged across multiple runs using CUDA Events.

Performance was evaluated using:

- Kernel execution time
- Achieved TFLOPS for GEMM
- Relative throughput against optimized implementations

## Benchmark Results

| Kernel | Custom Implementation | Reference | Relative Performance |
|---|---|---|---:|
| GEMM | 1.35 TFLOPS | cuBLAS: 3.56 TFLOPS | 37.9% |
| MLP | Custom CUDA | PyTorch | ~80–85% |
| FlashAttention | Custom CUDA | FlashAttention | ~70–85% |

The results demonstrate that progressively optimizing memory accesses and data
reuse can substantially narrow the performance gap with highly optimized
vendor libraries.

## GPU Optimization Principles

The project demonstrates several fundamental CUDA optimization principles:

### Memory Coalescing

Consecutive threads access consecutive memory locations, improving effective
global-memory bandwidth and reducing memory transactions.

### Shared-Memory Tiling

Matrix tiles are loaded cooperatively into shared memory and reused across
multiple computations, increasing arithmetic intensity and reducing global
memory traffic.

### Register Blocking

Each thread computes multiple output elements while keeping intermediate
accumulations in registers, increasing data reuse and instruction-level
parallelism.

### Warp Tiling

Work is organized at the warp level so threads collaborate on fixed output
tiles, improving workload distribution and GPU resource utilization.

### Online Softmax

FlashAttention maintains running normalization statistics while processing
attention tiles, avoiding storage of the complete attention matrix while
preserving numerical correctness.

## Repository Structure

A typical organization for the project is:

```text
.
├── gemm/
│   └── CUDA GEMM implementations
├── mlp/
│   └── CUDA MLP implementation
├── flashattention/
│   └── CUDA FlashAttention implementation
├── benchmarks/
│   └── Benchmarking and evaluation scripts
└── README.md
```

The exact source layout may vary with the repository version.

## Running and Benchmarking

The implementations require a CUDA-capable NVIDIA GPU and a CUDA development
environment with `nvcc`.

A typical compilation flow for an individual CUDA source file is:

```bash
nvcc -O3 <source>.cu -o <executable>
```

Run the resulting executable to benchmark the corresponding kernel.

For meaningful comparisons, the benchmark methodology used in the project
should be followed:

1. Warm up the kernel several times.
2. Execute multiple timed iterations.
3. Measure GPU execution using CUDA Events.
4. Average the measured execution times.
5. Compare against the appropriate reference implementation.

## References

- NVIDIA, **CUDA C++ Programming Guide**, 2024.
- NVIDIA, **cuBLAS Library User Guide**, 2024.
- Tri Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, NeurIPS 2022.
- Tri Dao, **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**, ICLR 2024.
- Simon Boehm, **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance**, 2022.
- PyTorch Documentation.
- NVIDIA CUTLASS.

## Author

**Monil Manish Desai**

Department of Computer Science and Engineering  
Indian Institute of Technology Bombay