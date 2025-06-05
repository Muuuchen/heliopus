#include "vecadd.cuh"

namespace cuda_op
{

__global__ void vecadd(int n, float* A, float* B, float* C)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    int tid = threadIdx.x;
    if (blockDim.x * blockIdx.x + tid < n)
    {
        C[blockDim.x * blockIdx.x + tid] =
            A[blockDim.x * blockIdx.x + tid] + B[blockDim.x * blockIdx.x + tid];
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}
} // namespace cuda_op