#include "launch_utils.cuh"
#include "rmsnorm.cuh"
#include <cstddef>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
int main()
{
    cudaStream_t stream = 0;
    constexpr int m = 1024;
    constexpr int n = 512;
    const size_t shmem_size = 48;
    float input[m * n];
    float output[m * n];
    float weight[m * n];
    //(512 + 31) / 32 + 31) / 32 * 32) =
    if (n % 8 == 0)
    {
        dim3 grid(m);
        dim3 block(std::min<int>(1024, ((n / 8 + 31) / 32 + 31) / 32 * 32));
        LAUNCH_KERNEL_WITH_PDL(cuda_op::rmsnorm_twoPassAlgo_e8, grid, block, shmem_size, stream,
                               static_cast<float4*>(static_cast<void*>(&output)),
                               static_cast<float4*>(static_cast<void*>(&input)),
                               static_cast<float4*>(static_cast<void*>(&weight)), m, n, 1e-5);
        LAUNCH_KERNEL_WITH_PDL(cuda_op::rmsnorm_twoPassAlgo_e8, grid, block, shmem_size, stream,
                               static_cast<float4*>(static_cast<void*>(&output)),
                               static_cast<float4*>(static_cast<void*>(&input)),
                               static_cast<float4*>(static_cast<void*>(&weight)), m, n, 1e-5);
        LAUNCH_KERNEL_WITH_PDL(cuda_op::rmsnorm_twoPassAlgo_e8, grid, block, shmem_size, stream,
                               static_cast<float4*>(static_cast<void*>(&output)),
                               static_cast<float4*>(static_cast<void*>(&input)),
                               static_cast<float4*>(static_cast<void*>(&weight)), m, n, 1e-5);
    }
}