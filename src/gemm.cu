// #include "cutlass/arch/mma.h"
// #include "cutlass/cutlass.h"
// #include "cutlass/epilogue/thread/linear_combination.h"
// #include "cutlass/gemm/device/gemm_splitk_parallel.h"
// #include "cutlass/gemm/device/gemm_universal.h"
// #include "cutlass/gemm/gemm_enumerated_types.h"
// #include "cutlass/gemm/threadblock/threadblock_swizzle.h"
// #include "cutlass/gemm_coord.h"
// #include "cutlass/half.h"
// #include "cutlass/numeric_size.h"
// #include "cutlass/util/device_memory.h"
// #include "gemm.cuh"

// #include "cutlass/util/command_line.h"
// #include "cutlass/util/host_tensor.h"
// #include "cutlass/util/reference/device/gemm.h"
// #include "cutlass/util/tensor_view_io.h"
// #include <cstddef>

// #define CUTLASS_CHECK(status)                                                                      \
//     {                                                                                              \
//         cutlass::Status error = status;                                                            \
//         if (error != cutlass::Status::kSuccess)                                                    \
//         {                                                                                          \
//             std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)                    \
//                       << "at: " << __LINE__ << std::endl;                                          \
//             exit(EXIT_FAILURE);                                                                    \
//         }                                                                                          \
//     }

// template <int ThreadBlockM, int ThreadBlockN, int ThreadBlockK, int WarpM, int WarpN, int WarpK,
//           int InstructionM, int InstructionN, int InstructionK, int NumStages, int SwizzleSize,
//           int SplitK>
// void cutlass_gemm_splitk(int M, int N, int K, const half* A, const half* B, half* C, half* D,
//                          cudaStream_t stream)
// {
//     using ThreadBlockShape = cutlass::gemm::GemmShape<ThreadBlockM, ThreadBlockN, ThreadBlockK>;
//     using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
//     using InstructionShape = cutlass::gemm::GemmShape<InstructionM, InstructionN, InstructionK>;
//     cutlass::gemm::GemmCoord problem_size(M, N, K); // problem shape is mnk GemmCoord

//     // Gemm A
//     using ElmentA = cutlass::half_t;
//     using LayoutA = cutlass::layout::RowMajor;
//     // 这里是用 TMA或者ldmatrix 的时候需要内存对齐，128位
//     constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElmentA>::value; // 128 / 16 = 8

//     // Gemm B
//     using ElmentB = cutlass::half_t;
//     using LayoutB = cutlass::layout::ColumnMajor;
//     constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElmentB>::value; // 128 / 16 = 8

//     // GEMM CD
//     using ElmentC = cutlass::half_t;
//     using LayoutC = cutlass::layout::RowMajor;
//     constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElmentC>::value; // 128 / 16 = 8

//     // and  some accumulators, precision  arch and tensorcore
//     using ElmentAccumulator = cutlass::half_t;
//     using ArchTag = cutlass::arch::Sm80;
//     using OperatorClass = cutlass::arch::OpClassTensorOp;

//     // Epilogue
//     using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//         ElmentC, AlignmentC, ElmentAccumulator,
//         ElmentAccumulator>; // 输出类型  每次访问的u元素数量 累加类型 计算类型

//     // classic data-parallel
//     using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
//         ElmentA, LayoutA, ElmentB, LayoutB, ElmentC, LayoutC, ElmentAccumulator, OperatorClass,
//         ArchTag, ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,
//         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleSize>, NumStages,
//         AlignmentA, AlignmentB>;

//     using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversal<
//         ElmentA, LayoutA, ElmentB, LayoutB, ElmentC, LayoutC, ElmentAccumulator, OperatorClass,
//         ArchTag, ThreadBlockShape, WarpShape, InstructionShape, EpilogueOp,
//         cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, NumStages, AlignmentA,
//         AlignmentB>;

//     // 再看看 example 06
//     // using DeviceGemmSplitK = cutlass::gemm::device::GemmSplitKParallel<typename ElementA_,
//     // typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename
//     // LayoutC_>
//     auto batch_stride_A = problem_size.m() * problem_size.k();
//     auto batch_stride_B = problem_size.k() * problem_size.n();
//     auto batch_stride_C = problem_size.m() * problem_size.n();
//     auto batch_stride_D = problem_size.m() * problem_size.n();

//     auto arguments = typename DeviceGemmBasic::Arguments(
//         cutlass::gemm::GemmUniversalMode::kGemm, problem_size, SplitK,
//         {
//             ElmentAccumulator(1.0f),
//             ElmentAccumulator(0.0f),

//         },
//         A, B, C, D, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D, K, // stride
//         a K,                                                                             //
//         stride b N, // stride c N);
//     DeviceGemmBasic gemm_op;
//     size_t workspace_size =
//         DeviceGemmBasic::get_workspace_size(arguments); // 执行矩阵运算所需的临时工作空间的字节数
//     cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
//     CUTLASS_CHECK(gemm_op.can_implement(arguments));
//     CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

//     CUTLASS_CHECK(gemm_op(stream));
// }
// // 仅显式实例化一个模板函数
// // 128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1)
// template void cutlass_gemm_splitk<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1>(
//     int M, int N, int K, const half* A, const half* B, half* C, half* D, cudaStream_t stream);
// // 分离模板
// int main() { return 0; }