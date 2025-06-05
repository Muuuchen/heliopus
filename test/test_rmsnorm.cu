#include <cuda_fp16.h>
#include <iostream>
#define FINAL_MASK 0xffffffff
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
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSum<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
    }
    warpReduceSum<T, NUM>(val);
    return (T)0.0f;
}
// 这里就是一个block算一行
__global__ void rmsnorm_twoPassAlgo_e8(float4* output, const float4* input, const float4* weight,
                                       const int m, const int n, float epsilon)
{
    /*Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

   */
    const int m_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_dim_x = blockDim.x;
    __shared__ float s_mean; // 这里一个block只用一个block 来存储mean， 看起来好是一个block算一行
    float local_sums[1] = {0.0f}; // 这样会在寄存器的吧【TODO】
    const int n_8 = n / 8;        // 这里把一行切成了8列
    int offset = m_idx * n_8; // 这里相当于把总量变成了八分，offset还是表达的是每行的第一个块的位置
    input += offset;
    output += offset;
// 这里block
// dim讲的是一个block中的线程数量，我用一个block中的每个线程来分别计算一个float4的平方和，那么每次都挪动一个线程快的数量
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    for (int index = tid; index < n_8; index += block_dim_x)
    {
        const float4 local_val = input[index];
        const half2* h1 = (half2*)&local_val.x;
        const half2* h2 = (half2*)&local_val.y;
        const half2* h3 = (half2*)&local_val.z;
        const half2* h4 = (half2*)&local_val.w;
        local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                         static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                         static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                         static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                         static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                         static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                         static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                         static_cast<float>(h4->y) * static_cast<float>(h4->y);
    }

    if (blockDim.x < 32)
    {
        warpReduceSum<float, 1>(local_sums);
    }
    else
    {
        blockReduceSum<float, 1>(local_sums);
    }
    if (threadIdx.x == 0)
    {
        s_mean = rsqrtf(local_sums[0] / n + epsilon);
    }
    __syncthreads();
    // 以上已经算完了RMS

    for (int index = tid; index < n_8; index += block_dim_x)
    {
        const float4 local_val = input[index];
        const float4 weight_val = weight[index];

        const half2* l1 = (half2*)&local_val.x;
        const half2* l2 = (half2*)&local_val.y;
        const half2* l3 = (half2*)&local_val.z;
        const half2* l4 = (half2*)&local_val.w;

        const half2* w1 = (half2*)&weight_val.x;
        const half2* w2 = (half2*)&weight_val.y;
        const half2* w3 = (half2*)&weight_val.z;
        const half2* w4 = (half2*)&weight_val.w;

        float4 tmp;
        half2* h1 = (half2*)&tmp.x;
        half2* h2 = (half2*)&tmp.y;
        half2* h3 = (half2*)&tmp.z;
        half2* h4 = (half2*)&tmp.w;
        h1->x = half(static_cast<float>(l1->x) * s_mean * static_cast<float>(w1->x));
        h1->y = half(static_cast<float>(l1->y) * s_mean * static_cast<float>(w1->y));
        h2->x = half(static_cast<float>(l2->x) * s_mean * static_cast<float>(w2->x));
        h2->y = half(static_cast<float>(l2->y) * s_mean * static_cast<float>(w2->y));
        h3->x = half(static_cast<float>(l3->x) * s_mean * static_cast<float>(w3->x));
        h3->y = half(static_cast<float>(l3->y) * s_mean * static_cast<float>(w3->y));
        h4->x = half(static_cast<float>(l4->x) * s_mean * static_cast<float>(w4->x));
        h4->y = half(static_cast<float>(l4->y) * s_mean * static_cast<float>(w4->y));
        output[index] = tmp;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

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
    dim3 grid(m);
    if (n % 8 == 0)
    {
        dim3 block(std::min(1024, (n / 8 + 31) / 32 * 32));
        rmsnorm_twoPassAlgo_e8<<<grid, block, 0, stream>>>(
            (float4*)d_output, (const float4*)d_input, (const float4*)d_weight, m, n, 1e-5);
        auto result = cudaGetLastError();
        if (result != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
            abort();
        }
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(
            cudaMemcpy(output, d_output, m * n * sizeof(half), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n * m; i++)
        {
            std::cout << (float)output[i] << std::endl;
        }
    }
}