# 设置CMake的最小版本要求
cmake_minimum_required(VERSION 3.0)

# 设置项目名称
project(trt_infer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 查找CUDA库
find_package(CUDA REQUIRED)

# 如果找到了CUDA，打印一些消息
if(CUDA_FOUND)
    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    libraries: ${CUDA_LIBRARIES}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
endif()

# 设置TensorRT库的路径
set(TENSORRT_ROOT "/home/lifa/Drivers/TensorRT-7.2.3.4")

# 将include目录加入到编译器的头文件搜索路径之下
# 添加TensorRT和CUDA头文件的搜索路径
include_directories(include ${CUDA_INCLUDE_DIRS} ${TENSORRT_ROOT}/include)

# 创建一个变量，包含所有的源文件
set(SOURCE_FILES trt_infer.cpp src/voxelization_trt.cpp)

# 指定生成目标
add_executable(trt_infer ${SOURCE_FILES})

# 链接库
target_link_libraries(trt_infer nvinfer nvinfer_plugin ${CUDA_LIBRARIES})
