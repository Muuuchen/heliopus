#ifndef REDUCE_CUH
#define REDUCE_CUH

#include <cuda_fp16.h>
#define FINAL_MASK 0xffffffff
/*模板函数（如 warpReduceSum 和
 * blockReduceSum）的声明和定义必须同时出现在头文件中，因为编译器需要在实例化时看到完整的函数体。*/
namespace cuda_op
{

// 这里需要搞清楚含义，相当于是对于一个输入数组，val[] 对NUM个元素，
// 每个元素都会讲warp的不同线程之间的相同元素进行规约
template <typename T, int NUM> __inline__ __device__ T warpReduceSum(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T)(0.0f);
}

// 这里主要是一个warp 规约不完所以需要在block 内进行 取决于我的block dim 的大小是不是大于32
template <typename T, int NUM> __inline__ __device__ T blockReduceSum(T* val)
{
    __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 31; // 0x11111 相当于对32 取余
    int wid = threadIdx.x >> 5;  // 相当于整除32
    warpReduceSum<T, NUM>(val);
    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }

        __syncthreads();
        // 这里接下里啊要对warp之后的值再reduce 需要mask
        bool is_mask = (threadIdx.x < (blockDim.x / 32));
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
        }
        warpReduceSum<T, NUM>(val);
    }
    return (T)0.0f;
}

} // namespace cuda_op
#endif