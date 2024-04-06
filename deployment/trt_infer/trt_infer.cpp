#include "include/common_trt.h"
#include "include/io_trt.hpp"
#include "include/voxelization_trt.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        // Suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// 简化错误处理
#define CHECK(status) \
    if (status != 0) \
    { \
        std::cerr << "Cuda failure: " << status << std::endl; \
        abort(); \
    }


void trt_infer(std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors,
               std::vector<int>& num_points_per_voxel){
    const std::string engineFilePath = "../../../pretrained/model.trt";  // 替换为你的 .trt 文件路径
    
    // 读取序列化的引擎
    auto engineData = readEngineFile(engineFilePath);
    
    // 支持插件(scatterND)
    // https://github.com/onnx/onnx-tensorrt/issues/597
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");

    // 创建运行时和引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 获取输入和输出的绑定索引
    int inputPillarsIndex = engine->getBindingIndex("input_pillars");
    int inputCoorsBatchIndex = engine->getBindingIndex("input_coors_batch");
    int inputNpointsPerPillarIndex = engine->getBindingIndex("input_npoints_per_pillar");
    int outputIndex = engine->getBindingIndex("output_x");

    // int numBindings = engine->getNbBindings(); // 获取绑定的数量
    // for (int i = 0; i < numBindings; ++i)
    // {
    //     const char* bindingName = engine->getBindingName(i); // 根据索引获取绑定名称
    //     std::cout << "Binding Index: " << i << ", Binding Name: " << bindingName << std::endl;
    // }

    // 使用CUDA分配设备内存
    int pillar_num = voxels.size();
    void* inputPillarsDevice;
    void* inputCoorsBatchDevice;
    void* inputNpointsPerPillarDevice;
    CHECK(cudaMalloc(&inputPillarsDevice, pillar_num * 32 * 4 * sizeof(float)));
    CHECK(cudaMalloc(&inputCoorsBatchDevice, pillar_num * 4 * sizeof(int)));
    CHECK(cudaMalloc(&inputNpointsPerPillarDevice, pillar_num * sizeof(int)));

    // 2d vector复制时需要注意, 分配临时主机内存
    Point* tempHostMemoryPillar = new Point[pillar_num * 32];
    Point* currentHostPtrPillar = tempHostMemoryPillar;
    
    // 一维化 Voxel 中的 Point 数据并复制到临时主机内存
    for (const auto& voxel : voxels) {
        memcpy(currentHostPtrPillar, voxel.points.data(), voxel.points.size() * sizeof(Point));
        currentHostPtrPillar += voxel.points.size();
    }
    
    // 2d vector复制时需要注意, 分配临时主机内存
    int* tempHostMemoryCoor = new int[pillar_num * 4];
    int* currentHostPtrCoor = tempHostMemoryCoor;
    
    // 一维化二维 vector 并复制到临时主机内存
    for (const auto& coor : coors) {
        memcpy(currentHostPtrCoor, coor.data(), coor.size() * sizeof(int));
        currentHostPtrCoor += coor.size();
    }

    // 将数据从主机复制到设备
    CHECK(cudaMemcpy(inputPillarsDevice, tempHostMemoryPillar, pillar_num * 32 * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(inputCoorsBatchDevice, tempHostMemoryCoor, pillar_num * 4 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(inputNpointsPerPillarDevice, num_points_per_voxel.data(), pillar_num * sizeof(int), cudaMemcpyHostToDevice));
    delete[] tempHostMemoryPillar;
    delete[] tempHostMemoryCoor;
    // delete[] currentHostPtrPillar;
    // delete[] currentHostPtrCoor;

    // 分配输出设备内存
    void* outputDevice;
    int outputSize = 100 * (7 + 3 + 1); // 根据模型输出确定outputSize
    CHECK(cudaMalloc(&outputDevice, outputSize * sizeof(float))); 

    // 创建输入和输出数据缓冲区指针数组
    void* buffers[4];
    buffers[inputPillarsIndex] = inputPillarsDevice;
    buffers[inputCoorsBatchIndex] = inputCoorsBatchDevice;
    buffers[inputNpointsPerPillarIndex] = inputNpointsPerPillarDevice;
    buffers[outputIndex] = outputDevice;

    // 执行推理
    context->enqueueV2(buffers, 0, nullptr);

    // 如果需要，将输出数据从设备复制回主机
    float* outputHost = new float[outputSize];
    cudaMemcpy(outputHost, outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < outputSize; i++){
    //     std::cout << outputHost[i] << " ";
    //     if (i > 0 && i % 11 == 0){
    //         std::cout << std::endl;
    //     }
    // }

    // 释放设备内存
    cudaFree(inputPillarsDevice);
    cudaFree(inputCoorsBatchDevice);
    cudaFree(inputNpointsPerPillarDevice);
    cudaFree(outputDevice);

    // 释放主机上的输出数组
    delete[] outputHost;

    // 销毁context
    context->destroy();

}


int main() {
    // 读入数据
    std::vector<Point> points_ori, points; 
    std::string file_path = "../../../dataset/demo_data/val/000134.bin";
    bool read_data_ok = read_points(file_path, points_ori);
    if (!read_data_ok) return 0;
    point_cloud_filer(points_ori, points);

    std::vector<float> voxel_size = {0.16, 0.16, 4};
    std::vector<float> coors_range = {0, -39.68, -3, 69.12, 39.68, 1};
    int max_points = 32;
    int max_voxels = 40000;
    
    std::vector<Voxel> voxels;
    std::vector<std::vector<int>> coors;
    std::vector<int> num_points_per_voxel;

    voxelize(points, voxel_size, coors_range, max_points, max_voxels, voxels, coors, num_points_per_voxel);

    /* 
    // check function voxelize
    std::cout << "voxels size: " << voxels.size() << std::endl;
    std::cout << "coors size: " << coors.size() << std::endl;
    std::cout << "num_points_per_voxel size: " << num_points_per_voxel.size() << std::endl;

    // 输出结果
    for (const auto& voxel : voxels) {
        for (const auto& point : voxel.points) {
            std::cout << point.x << " " << point.y << " " << point.z << " " << point.feature << std::endl;
        }
    }
    for (const auto& coor : coors) {
        std::cout << coor[0] << " " << coor[1] << " " << coor[2] << std::endl;
    }
    for (const auto& num : num_points_per_voxel) {
        std::cout << num << std::endl;
    }
    */

    std::vector<std::vector<int>> padded_coors;
    pad_coors(coors, padded_coors);
    trt_infer(voxels, padded_coors, num_points_per_voxel);
    
    
    return 0;
}