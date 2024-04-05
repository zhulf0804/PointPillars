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

    // 将数据从主机复制到设备
    CHECK(cudaMemcpy(inputPillarsDevice, voxels.data(), pillar_num * 32 * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(inputCoorsBatchDevice, coors.data(), pillar_num * 4 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(inputNpointsPerPillarDevice, num_points_per_voxel.data(), pillar_num * sizeof(int), cudaMemcpyHostToDevice));

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