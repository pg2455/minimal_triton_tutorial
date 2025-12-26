import torch
import numpy as np
import timeit
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for matrix multiplication using tiled computation on a 2D grid.

    Computes C = A @ B^T where:
    - A is MxK matrix
    - B is NxK matrix (transposed during multiplication)
    - C is MxN output matrix

    Algorithm:
    1. Each block computes one tile of the output matrix C
    2. Block position (pid_m, pid_n) determines which tile to compute
    3. For each tile, slide windows across K dimension:
       - Load BLOCK_SIZE x BLOCK_SIZE chunks from A and B
       - Compute partial dot products
       - Accumulate results
    4. Store final accumulated result to output tile

    Args:
        a_ptr: Pointer to matrix A (M x K)
        b_ptr: Pointer to matrix B (N x K) - will be transposed
        c_ptr: Pointer to output matrix C (M x N)
        M: Number of rows in A and C
        N: Number of rows in B, columns in C
        K: Number of columns in A and B (reduction dimension)
        BLOCK_SIZE: Tile size for blocking (compile-time constant)

    Note:
        - Uses 2D grid: each block handles one (BLOCK_SIZE x BLOCK_SIZE) tile of C
        - Sliding window pattern: tl.advance() moves block pointers along K dimension
        - BLOCK_SIZE must fit in SRAM or kernel will hang
        - Uses tl.dot() for fused multiply-add operations
    """
    # A: MxN; B: NxK; C:MxN; B is transposed here
    # Each block outputs a grid of C. 
    # We make this grid by sliding windows over the rows in A and cols in B
    # For each block in A and B

    # Location: Where is the block located in the grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Mark the pointers
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(K, 1),
        offsets=(pid_m*BLOCK_SIZE, 0), 
        block_shape=(BLOCK_SIZE, BLOCK_SIZE), order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(N, K), strides=(K, 1),
        offsets=(pid_n*BLOCK_SIZE, 0), 
        block_shape=(BLOCK_SIZE, BLOCK_SIZE), order=(1, 0),
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m*BLOCK_SIZE, pid_n*BLOCK_SIZE), 
        block_shape=(BLOCK_SIZE, BLOCK_SIZE), order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):

        # LOAD
        a_chunk = tl.load(a_block_ptr)
        b_chunk = tl.load(b_block_ptr)

        # DO MATH
        accumulator = tl.dot(a_chunk, b_chunk, accumulator) # fused multiply-add

        # Move the window 
        a_block_ptr = a_block_ptr.advance((0, BLOCK_SIZE))
        b_block_ptr = b_block_ptr.advance((0, BLOCK_SIZE))


    tl.store(c_block_ptr, accumulator)


def matmul(x: torch.Tensor, y: torch.Tensor):
    """
    CPU launcher for matrix multiplication kernel.

    Performs C = X @ Y^T using Triton's tiled matrix multiplication kernel.

    Args:
        x: First input matrix of shape (M, K), must be on CUDA device
        y: Second input matrix of shape (N, K), must be on CUDA device
           Note: y is treated as transposed, so effective computation is x @ y.T

    Returns:
        torch.Tensor: Output matrix of shape (M, N)

    Example:
        >>> M, N, K = 4096, 4096, 4096
        >>> x = torch.randn(M, K, device='cuda')
        >>> y = torch.randn(N, K, device='cuda')
        >>> result = matmul(x, y)  # shape: (4096, 4096)

    Performance:
        - Uses 2D grid of blocks, each computing a tile of the output
        - BLOCK_SIZE=64 balances SRAM usage and parallelism
        - Larger matrices benefit more from tiled computation
    """

    M = x.shape[0]
    N = y.shape[0]
    K = x.shape[1]
    out = torch.empty((M, N), device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE']), 
        triton.cdiv(N, meta['BLOCK_SIZE'])
    )

    matmul_kernel[grid](x, y, out, M, N, K, BLOCK_SIZE=64)

    return out

M, N, K = 4096, 4096, 4096
x = torch.randn(M, K, device='cuda')
y = torch.randn(N, K, device='cuda')
latency = triton.testing.do_bench(lambda : matmul(x, y))
print(f"Latency (ms): {latency}")