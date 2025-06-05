#include "launch_utils.cuh"
#include "rmsnorm.cuh"
#include "vecadd.cuh"
#include <cstddef>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#define CHECK_CUDA_ERROR(expr)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cudaError_t error = (expr);                                                                \
        if (error != cudaSuccess)                                                                  \
        {                                                                                          \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "                   \
                      << cudaGetErrorString(error) << std::endl;                                   \
            abort();                                                                               \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            std::cerr << "NO error" << __FILE__ << ":" << __LINE__ << ": "                         \
                      << cudaGetErrorString(error) << std::endl;                                   \
            ;                                                                                      \
        }                                                                                          \
    } while (0)

int main()
{
    cudaStream_t stream = 0;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream)); // 检查流创建错误（若使用非默认流）

    constexpr int m = 1024;
    constexpr int n = 512;
    const size_t shmem_size = 48;
    half input[m * n];
    half output[m * n];
    half weight[m * n];
    for (int i = 0; i < n * m; i++)
    {
        input[i] = 0.8f;
        output[i] = 0.0f;
        weight[i] = 1.1f;
    }
    half* d_input;
    half* d_output;
    half* d_weight;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, m * n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, m * n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight, m * n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, m * n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, output, m * n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight, weight, m * n * sizeof(half), cudaMemcpyHostToDevice));

    if (n % 8 == 0)
    {
        dim3 grid(m);
        dim3 block(std::min<int>(1024, (n / 8 + 31) / 32 * 32));
        std::cout << "compute:/n";
        LAUNCH_KERNEL_WITH_PDL(cuda_op::rmsnorm_twoPassAlgo_e8, grid, block, shmem_size, stream,
                               (float4*)d_output, (const float4*)(d_input),
                               (const float4*)(d_weight), m, n, 1e-5);
        CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误
        LAUNCH_KERNEL_WITH_PDL(cuda_op::rmsnorm_twoPassAlgo_e8, grid, block, shmem_size, stream,
                               (float4*)d_output, (const float4*)(d_input),
                               (const float4*)(d_weight), m, n, 1e-5);
        CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误
        LAUNCH_KERNEL_WITH_PDL(cuda_op::rmsnorm_twoPassAlgo_e8, grid, block, shmem_size, stream,
                               (float4*)d_output, (const float4*)(d_input),
                               (const float4*)(d_weight), m, n, 1e-5);
        CHECK_CUDA_ERROR(cudaGetLastError()); // 检查内核启动错误
        cudaDeviceSynchronize();
        std::cout << "done\n";
        cudaMemcpy(output, d_output, m * n * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++)
        {
            std::cout << (float)output[i] << " ";
        }
    }
}