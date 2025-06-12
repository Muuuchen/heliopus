cmake_minimum_required(VERSION 3.18)
project(pdl LANGUAGES CXX CUDA)
set(PROJECT_LIB_NAME pdl_core)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 启用C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90) 
# 生成compile_commands.json供clangd使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加头文件路径（Cutlass 和主项目头文件）
include_directories(
    ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
    ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include
    ${CMAKE_SOURCE_DIR}/include
)

# 启用CUDA支持
enable_language(CUDA)
find_package(CUDA REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# ===== 关键修改：将核心功能编译为静态库 =====
# 收集核心源文件（排除主入口文件 main.cu）
set(CORE_SOURCES
    src/rmsnorm.cu      
    src/distributed_gemm.cu
    src/gemm.cu
    src/pdl.cu
    src/vecadd.cu
)

# 生成静态库（名称为 pdl_core，类型为 STATIC）
add_library(${PROJECT_LIB_NAME} STATIC ${CORE_SOURCES})

# 配置CUDA编译选项（与主项目一致）
if(CMAKE_CUDA_COMPILER)
    set_target_properties(${PROJECT_LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(${PROJECT_LIB_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# ===== 主可执行文件（仅包含入口函数） =====
# 主可执行文件的源文件（仅包含 main.cu）
set(MAIN_EXE_SOURCES src/main.cu)

# 生成主可执行文件并链接静态库
add_executable(${PROJECT_NAME} ${MAIN_EXE_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LIB_NAME} cudart cuda)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
# 主可执行文件的CUDA编译选项（保持与库一致）
if(CMAKE_CUDA_COMPILER)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# 加载测试目录
add_subdirectory(test)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)