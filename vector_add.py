import torch
import triton
import numpy as np
import timeit
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for vector addition using automatic memory management with block pointers.

    This kernel demonstrates Triton's tl.make_block_ptr API for cleaner memory management:
    - No manual offset calculations needed
    - Triton handles pointer arithmetic automatically
    - Safer and more readable than manual pointer manipulation

    The kernel pattern:
    1. Get block position in grid (program ID)
    2. Create block pointers with automatic offset calculation
    3. Load data chunks from HBM
    4. Perform element-wise addition
    5. Store results back to HBM

    Args:
        a_ptr: Pointer to first input vector
        b_ptr: Pointer to second input vector
        c_ptr: Pointer to output vector
        N: Total number of elements in vectors
        BLOCK_SIZE: Number of elements processed per block (compile-time constant)

    Note:
        - Uses tl.make_block_ptr for automatic memory management
        - No manual masking needed - block pointers handle boundaries
        - Cleaner code compared to manual pointer arithmetic
    """
    # Find the program id of this particular block, i.e., position in the grid
    pid = tl.program_id(axis=0)

    # Memory pointers are managed by Triton
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(N,), strides=(1,),
        offsets=(pid*BLOCK_SIZE,), block_shape=(BLOCK_SIZE,), order=(0,),
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(N,), strides=(1,),
        offsets=(pid*BLOCK_SIZE,), block_shape=(BLOCK_SIZE,), order=(0,)
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(N,), strides=(1,),
        offsets=(pid*BLOCK_SIZE,), block_shape=(BLOCK_SIZE,), order=(0,)
    )

    # Load the data from HBM on to the thread registers
    a_chunk = tl.load(a_block_ptr)
    b_chunk = tl.load(b_block_ptr)

    # Do Math
    c_chunk = a_chunk + b_chunk

    # Store this data back in the HBM
    tl.store(c_block_ptr, c_chunk)


def vector_add(x: torch.Tensor, y: torch.Tensor):
    """
    CPU launcher for vector addition kernel with automatic memory management.

    Allocates output buffer and launches the Triton kernel with appropriate grid size.
    Compares performance against PyTorch's built-in addition operations.

    Args:
        x: First input tensor (1D, on CUDA device)
        y: Second input tensor (1D, same shape as x, on CUDA device)

    Returns:
        torch.Tensor: Element-wise sum of x and y

    Example:
        >>> N = 2**21
        >>> x = torch.randn(N, device='cuda')
        >>> y = torch.randn(N, device='cuda')
        >>> result = vector_add(x, y)
    """
    output = torch.zeros_like(x)
    N = x.shape[0]

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    
    vector_add_kernel[grid](x, y, output, N, BLOCK_SIZE=1024)
    return output

N = 2**21
x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')

latency = triton.testing.do_bench(lambda: vector_add(x,y)) # outputs ms
print(f"Latency (ms): {latency}")

times = []
for _ in range(10):
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    torch.cuda.synchronize() 
    start = timeit.default_timer()
    o = x + y
    torch.cuda.synchronize()
    times.append(timeit.default_timer() - start)

print(f"Manual time benchmark (without out tensor): {np.mean(times) * 1000: 0.4f}")


out = torch.empty_like(x, device='cuda')
times = []
for _ in range(10):
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    torch.cuda.synchronize() 
    start = timeit.default_timer()
    torch.add(x, y, out=out)
    torch.cuda.synchronize()
    times.append(timeit.default_timer() - start)

print(f"Manual time benchmark (with out tensor): {np.mean(times) * 1000: 0.4f}")
