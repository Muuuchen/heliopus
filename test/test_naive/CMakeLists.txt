get_filename_component(TEST_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
file(GLOB TEST_SOURCES *.cu *.cpp *.c *.cxx)
set(CMAKE_CUDA_ARCHITECTURES "90a")
if(NOT TEST_SOURCES)
    message(WARNING "No sources found in test: ${TEST_NAME}")
    return()
endif()

add_executable(${TEST_NAME} ${TEST_SOURCES})
target_link_libraries(${TEST_NAME} PRIVATE 
    pdl_core
    cudart 
    cuda
)

# +++ 确保测试程序也输出到 bin 目录 +++
set_target_properties(${TEST_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
    ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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