#include "cuda_runtime.h"

__global__ void copy1(float* __restrict__ a, float* __restrict__ b, int32_t n)
{
    float tmp = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        tmp += b[i];
    }
    // Initial work that should finish before starting secondary kernel
    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        b[i] = a[i] + tmp;
    }
}

__global__ void copy2(const float* __restrict__ b, float* __restrict__ c,
                      const float* __restrict__ d, int32_t n)
{
    float result = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        result += d[i];
    }
    // 这部分对于第一个kernel
    // 没有依赖，启动后执行到cudaGridDependencySynchronize()的时候才会阻塞，等待第一个kernel flush
    // memory Independent work
    // Will block until all primary kernels the secondary kernel is dependent on have completed and
    // flushed results to global memory
    cudaGridDependencySynchronize();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        c[i] = b[i] + result + 2.0f;
    }
}

void pdl_main()
{
    const int n = 4096 * 4096;
    // const int n = 64 * 4096;

    float *a, *b, *c, *d;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    cudaMalloc(&d, n * sizeof(float));

    const int32_t grid = 16;
    const int32_t blockdim = 256;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < 10; i++)
    {
        cudaLaunchConfig_t kernelConfig2 = {0};
        kernelConfig2.gridDim = grid;
        kernelConfig2.blockDim = blockdim;
        kernelConfig2.dynamicSmemBytes = 0;
        kernelConfig2.stream = stream;
        cudaLaunchAttribute attribute2[1];
        attribute2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute2[0].val.programmaticStreamSerializationAllowed = 1;
        kernelConfig2.attrs = attribute2;
        kernelConfig2.numAttrs = 1;

        // PDL Launch
        copy1<<<grid, blockdim, 0, stream>>>(a, b, n);
        cudaLaunchKernelEx(&kernelConfig2, &copy2, b, c, d, n);

        // No PDL Launch
        // copy1<<<grid, blockdim, 0, stream>>>(a, b, n);
        // copy2<<<grid, blockdim, 0, stream>>>(b, c, d, n);
    }

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);

    cudaStreamDestroy(stream);
}
