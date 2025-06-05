
#include <cuda_fp16.h>

namespace cuda_op
{

__global__ void vecadd(int n, float* A, float* B, float* C);
} // namespace cuda_op