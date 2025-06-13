#pragma once

#include <cstdint>
#include <cuda.h>

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_ext.hpp>

// stmatrix.sync.aligned.x2.m8n8.shared.b16
// Collectively store one or more matrices to shared memory.
//在一个线程束的所有线程中，将一个或多个矩阵集中存储到共享状态空间中地址操作数 p 所指示的位置
template <typename dtype_t> struct SM90_U32x2_STSM_N
{
    __device__ __forceinline__ static void copy(dtype_t src_0, dtype_t src_1, void* smem_dst)
    {
        // 这里的src应该是寄存器文件, 进行地址转换
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0),
                                 *reinterpret_cast<uint32_t*>(&src_1)};
        // sync 的同步
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"l"(smem_dst),
                     "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t> struct SM90_U32x4_STSM_N
{
    __device__ __forceinline__ static void copy(dtype_t src_0, dtype_t src_1, dtype_t src_2,
                                                dtype_t src_3, void* smem_dst)
    {
        const uint32_t src[4] = {
            *reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
            *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile(
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(smem_dst),
            "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

//异步代理会记录任务参数（如输入矩阵的地址、计算精度、输出位置），并将任务加入队列，无需等待计算完成。
/*
调用mma_async时，异步代理会记录任务参数（如输入矩阵的地址、计算精度、输出位置），并将任务加入队列，无需等待计算完成。
GPU 计算单元可以继续执行后续指令（如加载下一批数据、执行其他计算）。
当需要结果时（如后续指令依赖该矩阵乘加的输出），通过mma_sync指令通知异步代理：等待当前任务完成，并将结果写入目标寄存器或内存。
*/

// : "memory"
// 告诉编译器这段汇编可能会修改内存（或依赖内存状态），强制编译器刷新缓存、禁用相关内存优化，避免因编译器优化导致的内存访问不一致问题。
__forceinline__ __device__ void warpgroup_arrive()
{
    // wgmma.fence 操作用于表明跨线程束组（warpgroup）的寄存器 / 共享内存已被写入。然后可以用于计算
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__forceinline__ __device__ void warpgroup_commit_batch()
{
    //将之前所有未提交的 wgmma.mma_async 操作提交到一个 wgmma 组中。
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N> __forceinline__ __device__ void warpgroup_wait()
{
    // wgmma.wait_group 指令将导致执行线程等待，直到最近的 wgmma 组中只有 N
    // 个或更少的组处于未完成状态，并且执行线程提交的所有先前 wgmma 组都已完成。
    // 例如，当 N 为 0 时，执行线程等待所有先前的 wgmma 组完成。操作数 N 是一个整数常量

    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__forceinline__ __device__ void warpgroup_fence_operand(float& reg)
{
    asm volatile("" : "+f"(reg) : : "memory");
    // 这里的汇编体为空，主要是为了告诉编译器不要优化这段代码
    // 通过 "+f"(reg) 告诉编译器 reg 是一个 float 类型的寄存器变量，并且需要读写
    // "memory" 告诉编译器这段代码可能会修改内存状态，强制刷新寄存器缓存
    // 这样可以确保在多线程环境下，reg
    // 的值在栅栏点前后保持一致，不会因为编译器优化导致内存访问不一致
}

__forceinline__ __device__ uint32_t get_lane_id()
{
    //%laneid：是 CUDA 内联汇编中的硬件特殊寄存器，由 GPU 硬件和 CUDA 编译器共同定义。它是 GPU
    //硬件为每个线程内置的状态寄存器，
    uint32_t lane_id;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}


