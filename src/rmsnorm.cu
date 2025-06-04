#include "rmsnorm.cuh"
namespace cuda_op
{
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
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
}
} // namespace cuda_op