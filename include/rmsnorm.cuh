#ifndef RMSNORM_CUH
#define RMSNORM_CUH
#include "reduce.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// if normalized_shape is (3, 5) (a 2-dimensional shape), the RMS is computed over the last 2
// dimensions of the input.
namespace cuda_op
{
// 这里就是一个block算一行
__global__ void rmsnorm_twoPassAlgo_e8(float4* output, const float4* input, const float4* weight,
                                       const int m, const int n, float epsilon);
} // namespace cuda_op
#endif