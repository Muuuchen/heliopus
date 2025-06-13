// #pragma once

// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wunknown-attributes"

// #include <cute/swizzle.hpp>
// #include <cute/arch.hpp>
// #include <cute/arch/cluster_sm90.hpp>
// #include <cute/arch/copy_sm90.hpp>

#include "cute/arch/copy_sm90_desc.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 自定义内存描述符（示例）
struct GmmaDescriptor
{
    uint32_t start_address; // 共享内存起始地址（转换后的值）
    uint32_t stride;        // 内存步长
};

// 内核函数：将共享内存指针转换为描述符
__global__ void kernel(float* device_ptr)
{
    // 1. 声明共享内存（1KB，假设用于存储矩阵数据）
    __shared__ float shared_mem[256]; // 256 * 4 bytes = 1KB

    // 2. 获取共享内存的通用指针（需显式转换为void*）
    void* generic_ptr = static_cast<void*>(shared_mem);

    // 3. 使用 __cvta_generic_to_shared 转换为共享内存地址（uint32_t）
    uint32_t shared_address = __cvta_generic_to_shared(generic_ptr);

    // 4. 填充自定义内存描述符（示例逻辑）
    GmmaDescriptor desc;
    desc.start_address = shared_address;
    desc.stride = sizeof(float) * 16; // 假设步长为16个float元素

    // （可选）验证：将共享内存数据复制到全局内存（仅演示用）
    shared_mem[threadIdx.x] = static_cast<float>(threadIdx.x); // 初始化共享内存
    __syncthreads(); // 同步线程，确保数据写入完成

    // 通过转换后的地址间接访问共享内存（实际中可能通过描述符操作）
    float* shared_ptr = reinterpret_cast<float*>(shared_address);
    device_ptr[threadIdx.x] = shared_ptr[threadIdx.x];
}

int main()
{
    const int N = 256;
    float* device_ptr;
    cudaMalloc(&device_ptr, N * sizeof(float));
    float* host_ptr = new float[N];
    cute::prefetch_tma_descriptor(nullptr);
    // 启动内核（1个线程块，256个线程）
    kernel<<<1, N>>>(device_ptr);
    cudaDeviceSynchronize();

    // 拷贝结果到主机并打印前10个元素
    cudaMemcpy(host_ptr, device_ptr, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Shared memory content (first 10 elements):\n");
    for (int i = 0; i < 10; ++i)
    {
        printf("%.1f ", host_ptr[i]);
    }
    printf("\n");

    // 清理内存
    cudaFree(device_ptr);
    delete[] host_ptr;
    return 0;
}
