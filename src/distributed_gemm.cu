// (base) root@VM-162-200-tencentos:~/workspace/overlap# nvidia-smi topo -m
//         GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA
//         Affinity   GPU NUMA ID
// GPU0     X      NODE    NODE    NODE    SYS     SYS     SYS     SYS     0-191   0 N/A GPU1 NODE
// X      PIX     NODE    SYS     SYS     SYS     SYS     0-191   0               N/A GPU2    NODE
// PIX      X      NODE    SYS     SYS     SYS     SYS     0-191   0               N/A GPU3    NODE
// NODE    NODE     X      SYS     SYS     SYS     SYS     0-191   0               N/A GPU4    SYS
// SYS     SYS     SYS      X      NODE    NODE    NODE    192-383 1               N/A GPU5    SYS
// SYS     SYS     SYS     NODE     X      PIX     NODE    192-383 1               N/A GPU6    SYS
// SYS     SYS     SYS     NODE    PIX      X      NODE    192-383 1               N/A GPU7    SYS
// SYS     SYS     SYS     NODE    NODE    NODE     X      192-383 1               N/A

// Legend:

//   X    = Self
//   SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g.,
//   QPI/UPI) NODE = Connection traversing PCIe as well as the interconnect between PCIe Host
//   Bridges within a NUMA node PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
//   (typically the CPU) PXB  = Connection traversing multiple PCIe bridges (without traversing the
//   PCIe Host Bridge) PIX  = Connection traversing at most a single PCIe bridge NV#  = Connection
//   traversing a bonded set of # NVLinks

#include <cuda.h>
#include <cuda_runtime.h>
// #include <nvshmemx.h>

#include <nccl.h>