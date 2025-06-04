cmake_minimum_required(VERSION 3.18)
project(OverLap LANGUAGES CXX CUDA)

# 启用C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90) 
# 生成compile_commands.json供clangd使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 添加cutlass头文件路径
include_directories(
    # ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
    # ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include
    ${CMAKE_SOURCE_DIR}/include
)

# 如果你的代码需要CUDA支持
enable_language(CUDA)
find_package(CUDA REQUIRED)

# 添加可执行文件
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
add_executable(${PROJECT_NAME} ${SOURCES})

# 设置CUDA编译选项
if(CMAKE_CUDA_COMPILER)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# 安装规则(可选)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)