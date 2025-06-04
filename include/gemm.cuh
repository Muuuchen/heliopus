#ifndef GEMM_CUH
#define GEMM_CUH
#include <cuda_fp16.h>

template <int ThreadBlockM, int ThreadBlockN, int ThreadBlockK, int WarpM, int WarpN, int WarpK,
          int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize,
          int SplitK>
void cutlass_gemm_splitk(int M, int N, int K, const half* A, const half* B, half* C, half* D,
                         cudaStream_t stream);

#endif