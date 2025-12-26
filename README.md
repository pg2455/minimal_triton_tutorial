# Minimal Triton Tutorial

A hands-on tutorial for learning GPU programming with Triton through three progressively complex examples. This repository contains code examples that demonstrate how to write efficient CUDA kernels using Triton's Python-like syntax.

For a detailed explanation of the concepts, hardware model, and programming patterns, read the full blog post: [www.pgupta.info/blog/2025/12/triton/](https://www.pgupta.info/blog/2025/12/triton/)

## What is Triton?

Triton is a language and compiler for writing highly efficient custom GPU kernels. Unlike CUDA, which requires low-level programming, Triton lets you write GPU code that looks like Python while achieving performance comparable to hand-written CUDA.

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8 or later
- CUDA Toolkit installed

## Installation

Install the required packages:

```bash
pip install torch triton numpy
```

For the latest Triton version:
```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## Repository Structure

```
.
├── add.py           # Example 1: Element-wise addition with manual memory management
├── vector_add.py    # Example 2: Vector addition with automatic memory management
├── matmul.py        # Example 3: Matrix multiplication with 2D tiling
└── README.md        # This file
```

## Examples

### Example 1: Element-wise Addition (`add.py`)

**Purpose**: Learn the fundamentals of Triton programming with manual memory management

**Key Concepts**:
- Manual pointer arithmetic and offset calculations
- Program ID and block-based parallelism
- Masking for boundary conditions
- Basic kernel launch patterns

**Run**:
```bash
python add.py
```

**What it demonstrates**:
- How to write a `@triton.jit` decorated kernel function
- Manual calculation of memory offsets using `tl.program_id()` and `tl.arange()`
- Loading data from GPU memory (HBM) to registers with `tl.load()`
- Storing results back with `tl.store()`
- Grid-based kernel launching

### Example 2: Vector Addition (`vector_add.py`)

**Purpose**: Learn automatic memory management using Triton's block pointer API

**Key Concepts**:
- `tl.make_block_ptr()` for cleaner memory management
- Automatic pointer arithmetic
- Performance comparison with PyTorch operations
- Impact of memory allocation on performance

**Run**:
```bash
python vector_add.py
```

**What it demonstrates**:
- Simplified memory management compared to manual pointer arithmetic
- How block pointers handle offset calculations automatically
- Performance comparison: Triton vs PyTorch native operations
- Impact of pre-allocating output tensors on performance

### Example 3: Matrix Multiplication (`matmul.py`)

**Purpose**: Implement tiled matrix multiplication using 2D grids

**Key Concepts**:
- 2D grid programming with `tl.program_id(0)` and `tl.program_id(1)`
- Tiled computation for large matrices
- Sliding window pattern with `advance()`
- SRAM constraints and block size considerations

**Run**:
```bash
python matmul.py
```

**What it demonstrates**:
- Multi-dimensional grid layouts
- Tiled algorithm for efficient matrix multiplication
- How to use `tl.dot()` for fused multiply-add operations
- Accumulator pattern for reduction operations
- Memory hierarchy considerations (SRAM vs HBM)

## Understanding the Output

All examples use `triton.testing.do_bench()` for performance measurement, which:
- Flushes the L2 cache between runs for accurate "cold start" measurements
- Runs multiple iterations (warmup + measurement)
- Reports latency in milliseconds

The vector_add.py example also includes manual benchmarks to show:
1. PyTorch overhead when allocating output tensors
2. Performance parity when output is pre-allocated

## Performance Tips

1. **Block Size**: Must fit in SRAM (~164-228 KB per block). Too large will cause the kernel to hang.
2. **Memory Coalescing**: Adjacent threads should access adjacent memory locations for optimal bandwidth.
3. **Autotuning**: Use `@triton.autotune` to automatically find the best block size and warp configuration.
4. **Cache Effects**: Real-world performance may differ from benchmarks due to L2 cache hits.

## Common Issues

**"Kernel hangs" or "No output"**: Your BLOCK_SIZE is too large for the available SRAM. Try smaller values (64, 128, 256).

**"CUDA out of memory"**: Reduce the problem size (N, M, K values) or use smaller batch sizes.

**"No CUDA device found"**: Ensure you have a CUDA-capable GPU and correct drivers installed. Some examples include a `RUN_ON_CPU` flag for testing (much slower).

## Learning Path

1. Start with `add.py` to understand the basic Triton programming model
2. Move to `vector_add.py` to learn automatic memory management
3. Study `matmul.py` to understand multi-dimensional grids and tiling

For deeper understanding of:
- GPU memory hierarchy (HBM, L2, SRAM, registers)
- CUDA compute layout (SMs, warps, threads)
- Memory coalescing and bandwidth optimization
- Hardware-software mapping

Read the full tutorial: [BLOG_URL]

## Contributing

Found an issue or have suggestions? Please open an issue or submit a pull request!

## License

MIT License - feel free to use this code for learning and teaching.

## Additional Resources

- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
