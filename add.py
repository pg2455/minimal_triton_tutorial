import torch
import triton
import triton.language as tl
import os 

RUN_ON_CPU = False 

## CPU FLAG 
if RUN_ON_CPU:
    os.environ['TRITON_INTERPRET'] = "1"
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

# 1. THE KERNEL (Runs on GPU)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for element-wise vector addition using manual memory management.

    This kernel demonstrates the fundamental pattern of Triton programming:
    1. Calculate which block this is (program ID)
    2. Calculate memory offsets for this block
    3. Load data from HBM to registers
    4. Perform computation
    5. Store results back to HBM

    Args:
        x_ptr: Pointer to the first input tensor in GPU memory
        y_ptr: Pointer to the second input tensor in GPU memory
        output_ptr: Pointer to the output tensor in GPU memory
        n_elements: Total number of elements in the input tensors
        BLOCK_SIZE: Number of elements each block processes (compile-time constant)

    Note:
        - Uses manual pointer arithmetic with offsets
        - Masking prevents out-of-bounds memory access
        - Each block processes BLOCK_SIZE elements in parallel
    """
    # Get the program ID (like a thread ID)
    pid = tl.program_id(axis=0)
    
    # Create offsets (The "Spatial" logic)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds access
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Do math
    output = x + y
    
    # Store data
    tl.store(output_ptr + offsets, output, mask=mask)

# 2. THE LAUNCHER (Runs on CPU)
def add(x: torch.Tensor, y: torch.Tensor):
    """
    CPU wrapper function that launches the Triton add kernel on the GPU.

    This function handles:
    - Output tensor allocation
    - Grid size calculation (how many blocks to launch)
    - Kernel launch with appropriate parameters

    Args:
        x: First input tensor (must be on GPU)
        y: Second input tensor (must be on GPU, same shape as x)

    Returns:
        torch.Tensor: Element-wise sum of x and y

    Example:
        >>> x = torch.rand(1000, device='cuda')
        >>> y = torch.rand(1000, device='cuda')
        >>> result = add(x, y)
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Define the Grid (How many workers?)
    # We need enough blocks to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch!
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# 3. TEST IT
x = torch.rand(1000, device=device)
y = torch.rand(1000, device=device)
# output = add(x, y)
# print(output)

latency_ms = triton.testing.do_bench(lambda : add(x, y), warmup=25, rep=100)
print(f"Latency: {latency_ms} ms")