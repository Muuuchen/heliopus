# 获取当前测试目录名作为可执行文件名
get_filename_component(TEST_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 自动收集所有源文件
file(GLOB TEST_SOURCES *.cu *.cpp *.c *.cxx)

if(NOT TEST_SOURCES)
    message(WARNING "No sources found in test: ${TEST_NAME}")
    return()
endif()

message(STATUS "Configuring test: ${TEST_NAME}")

# 创建测试可执行文件
add_executable(${TEST_NAME} ${TEST_SOURCES})

# 链接核心库和CUDA依赖
target_link_libraries(${TEST_NAME} PRIVATE 
    pdl_core
    cudart 
    cuda
)

# 包含路径 (继承根设置 + 当前目录)
target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
    ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}  # 包含测试目录自身
)

# CUDA特定设置
if(CMAKE_CUDA_COMPILER)
    target_compile_options(${TEST_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# 可选：添加为CTest测试
# add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})