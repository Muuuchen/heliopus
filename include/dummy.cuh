#define LAUNCH_KERNEL_WITH_PDL(kernel, grid_dim, block_dim, smem_size, stream, ...)                                     \Add commentMore actions
    do {                                                                                                                \
        int device__;                                                                                                   \
        cudaGetDevice(&device__);                                                                                       \
        int cc_major__;                                                                                                 \
        cudaDeviceGetAttribute(&cc_major__, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, device__);               \
        if (cc_major__ >= 9) {                                                                                          \
            cudaLaunchConfig_t config__;                                                                                \
            config__.gridDim          = (grid_dim);                                                                     \
            config__.blockDim         = (block_dim);                                                                    \
            config__.dynamicSmemBytes = (smem_size);                                                                    \
            config__.stream           = (stream);                                                                       \
            cudaLaunchAttribute attrs__[1];                                                                             \
            attrs__[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization; \
            attrs__[0].val.programmaticStreamSerializationAllowed = rtp_llm::getEnvEnablePDL();                         \
            config__.numAttrs                                     = 1;                                                  \
            config__.attrs                                        = attrs__;                                            \
            cudaLaunchKernelEx(&config__, &(kernel), __VA_ARGS__);                                                      \
        } else {                                                                                                        \
            (kernel)<<<(grid_dim), (block_dim), (smem_size), (stream)>>>(__VA_ARGS__);                                  \
        }                                                                                                               \
    } while (0)