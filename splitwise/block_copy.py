import torch
from torch.utils.cpp_extension import load_inline
import os

cuda_arch = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch


def get_block_copy(num_block, block_size, num_thd=128):
    cuda = f"""
#include <ATen/cuda/CUDAContext.h>
constexpr int num_ptr = {num_block};
constexpr int block_size = {block_size};
constexpr int num_thd = {num_thd};

struct ptr_arr {{
    float4* __restrict__ ptrs[num_ptr];
}};

__global__ void 
__launch_bounds__(num_thd)
gather_kernel(float4* __restrict__ dst, const ptr_arr src) {{
    __builtin_assume(block_size > 0);
    __builtin_assume(block_size % num_thd == 0);
    int blk_id = blockIdx.x;
    auto src_ptr = src.ptrs[blk_id];
    auto dst_ptr = &dst[blk_id * block_size];
    int tid = threadIdx.x;
    #pragma unroll (block_size / num_thd)
    for (int i=tid; i < block_size; i += num_thd) {{
        dst_ptr[i] = src_ptr[i];
    }}
}}

__global__ void 
__launch_bounds__(num_thd)
scatter_kernel(ptr_arr dst, const float4* __restrict__ src) {{
    __builtin_assume(block_size > 0);
    __builtin_assume(block_size % num_thd == 0);
    int blk_id = blockIdx.x;
    auto src_ptr = &src[blk_id * block_size];
    auto dst_ptr = dst.ptrs[blk_id];
    int tid = threadIdx.x;
    #pragma unroll (block_size / num_thd)
    for (int i=tid; i < block_size; i += num_thd) {{
        dst_ptr[i] = src_ptr[i];
    }}
}}

void gather(at::Tensor& dst, std::vector<at::Tensor> src) {{
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, dst.type(), "gather", ([&] {{
        ptr_arr src_ptrs;
        auto dst_ptr = (float4*)dst.data_ptr<scalar_t>();
        for (int i=0; i< num_ptr; i++) {{
            src_ptrs.ptrs[i] = (float4*)src[i].data_ptr<scalar_t>();
        }}
        static_assert(block_size % num_thd == 0);
        static_assert(block_size % 16 == 0);
        gather_kernel<<<num_ptr, num_thd, 0, at::cuda::getCurrentCUDAStream()>>>(dst_ptr, src_ptrs);
    }}));
}}

void scatter(std::vector<at::Tensor> dst, at::Tensor& src) {{
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, src.type(), "scatter", ([&] {{
        ptr_arr dst_ptrs;
        auto src_ptr = (float4*)src.data_ptr<scalar_t>();
        for (int i=0; i< num_ptr; i++) {{
            dst_ptrs.ptrs[i] = (float4*)dst[i].data_ptr<scalar_t>();
        }}
        static_assert(block_size % num_thd == 0);
        static_assert(block_size % 16 == 0);
        scatter_kernel<<<num_ptr, num_thd, 0, at::cuda::getCurrentCUDAStream()>>>(dst_ptrs, src_ptr);
    }}));
}}
    """
    cpp = """
        void gather(at::Tensor& dst, std::vector<at::Tensor> src);
        void scatter(std::vector<at::Tensor> dst, at::Tensor& src);
        """
    return load_inline(
        "block_copy",
        cpp_sources=[cpp],
        cuda_sources=[cuda],
        functions=["gather", "scatter"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
    )


if __name__ == "__main__":
    dtype = torch.bfloat16
    num_block = 64
    block_size = 16 * 8 * 128 * dtype.itemsize // 16
    src = torch.empty(2, 1024, 16, 8, 128, device="cuda", dtype=dtype)

    for i in range(1024):
        src[0, i, ...] = i * 2
        src[1, i, ...] = i * 2 + 1

    dst = torch.empty(
        (num_block, 16, 8, 128), device="cuda", dtype=dtype
    )


    src_blocks = [src[0, i, ...] for i in range(1, num_block, 2)] + [
        src[1, i, ...] for i in range(0, num_block, 2)
    ]

    assert len(src_blocks) == num_block

    block_copy = get_block_copy(num_block, block_size)

    block_copy.gather(dst, src_blocks)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for i in range(5):
        block_copy.gather(dst, src_blocks)
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end) / 5
    bandwidth = num_block * block_size * 16 / 2**30 / (elapsed_time_ms / 1000)
    print(f"[Gather] Elapsed time: {elapsed_time_ms:.3f} ms", f"Bandwidth: {bandwidth:.3f} GB/s")

    for i, j in enumerate(range(1, num_block, 2)):
        assert torch.allclose(dst[i], src[0, j, ...]), f"Failed at dst[{i}] {dst[i]} src[0, {j}, ...] {src[0, j, ...]}"
    for i, j in enumerate(range(0, num_block, 2)):
        i_ = i + num_block // 2
        assert torch.allclose(dst[i_], src[1, j, ...]), f"Failed at dst[{i_}] {dst[i]} src[1, {j}, ...] {src[1, j, ...]}"


    dst = dst * 2

    block_copy.scatter(src_blocks, dst)
    torch.cuda.synchronize()
    start.record()
    for i in range(5):
        block_copy.scatter(src_blocks, dst)
    end.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start.elapsed_time(end) / 5
    bandwidth = num_block * block_size * 16 / 2**30 / (elapsed_time_ms / 1000)
    print(f"[Scatter] Elapsed time: {elapsed_time_ms:.3f} ms", f"Bandwidth: {bandwidth:.3f} GB/s")

    for i in range(1, num_block, 2):
        ref = src[0, i, ...].new_full((16, 8, 128), (i * 2) * 2)
        assert torch.allclose(src[0, i, ...], ref), f"Failed at src[0, {i}, ...] {src[0, i, ...]} ref {ref}"
    for i in range(0, num_block, 2):
        ref = src[1, i, ...].new_full((16, 8, 128), (i * 2 + 1) * 2)
        assert torch.allclose(src[1, i, ...], ref), f"Failed at src[1, {i}, ...] {src[1, i, ...]} ref {ref}"


