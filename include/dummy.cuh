
namespace cuda_op
{
__global__ void dummy_kernel(int dummy)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    __syncthreads();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}
} // namespace cuda_op