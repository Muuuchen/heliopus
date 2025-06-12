# 自动发现所有测试子目录
file(GLOB TEST_SUBDIRS LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)
set(CMAKE_CUDA_ARCHITECTURES "90a")
foreach(test_dir ${TEST_SUBDIRS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/CMakeLists.txt)
        message(STATUS "Adding test directory: ${test_dir}")
        add_subdirectory(${test_dir})
    endif()
endforeach()