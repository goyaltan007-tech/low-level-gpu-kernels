import time
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

def add_lists_on_gpu(x, y):
    size = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, 1024),)
    
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
    return output

SIZE = 10_000_000
x_cpu = torch.rand(SIZE)
y_cpu = torch.rand(SIZE)

x_gpu = x_cpu.cuda()
y_gpu = y_cpu.cuda()

print("Calculating on CPU...")
start_cpu = time.time()
cpu_result = x_cpu + y_cpu
time_cpu = time.time() - start_cpu
print(f"CPU Time: {time_cpu:.5f} seconds")

print("\nCalculating on GPU...")
add_lists_on_gpu(x_gpu, y_gpu) 
torch.cuda.synchronize()       

start_gpu = time.time()
gpu_result = add_lists_on_gpu(x_gpu, y_gpu)
torch.cuda.synchronize()       
time_gpu = time.time() - start_gpu
print(f"GPU Time: {time_gpu:.5f} seconds")

speedup = time_cpu / time_gpu
print(f"\nSUCCESS: The GPU was {speedup:.1f}x FASTER than the CPU!")
