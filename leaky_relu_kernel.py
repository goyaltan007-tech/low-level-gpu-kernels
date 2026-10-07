import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(x_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask=mask)
    
    output = tl.where(x > 0.0, x, x * 0.01)

    tl.store(output_ptr + offsets, output, mask=mask)

def apply_leaky_relu_on_gpu(x):
    size = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, 1024),)
    
    leaky_relu_kernel[grid](x, output, size, BLOCK_SIZE=1024)
    return output

test_data = torch.tensor([-2.5, -1.0, 0.0, 3.5, 5.2], device='cuda', dtype=torch.float32)
gpu_result = apply_leaky_relu_on_gpu(test_data)

print("🚀 Live GPU Activation Completed!")
print("Inputs went in :", test_data.cpu().tolist())
print("Outputs came out:", gpu_result.cpu().tolist())
