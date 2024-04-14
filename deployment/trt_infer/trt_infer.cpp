#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <chrono>

#include "include/common_trt.h"
#include "include/io_trt.hpp"
#include "include/voxelization_trt.h"
#include "include/utils.h"
#include "include/nms_trt.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        switch(severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
            case Severity::kWARNING:
                std::cout << msg << std::endl;
                break;
            case Severity::kINFO:
            case Severity::kVERBOSE:
                // Optionally ignore or handle less severe messages
                break;
        }
    }
} gLogger;

// 简化错误处理
#define CHECK(status) \
    if (status != 0) \
    { \
        std::cerr << "Cuda failure: " << status << std::endl; \
        abort(); \
    }

std::pair<nvinfer1::ICudaEngine*, nvinfer1::IExecutionContext*> initializeTensorRTComponents(const std::string& engineFilePath) {
    // 支持插件(scatterND)
    // https://github.com/onnx/onnx-tensorrt/issues/597
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");

    // 读取序列化的引擎
    auto engineData = readEngineFile(engineFilePath);

    // 创建运行时和引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);

    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    return {engine, context};
}

void trtInfer(std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors, std::vector<int>& num_points_per_voxel, 
              int& max_points, nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, 
              std::vector<float>& output){

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
    CHECK(cudaMalloc(&inputPillarsDevice, pillar_num * max_points * sizeof(Point)));
    CHECK(cudaMalloc(&inputCoorsBatchDevice, pillar_num * 4 * sizeof(int)));
    CHECK(cudaMalloc(&inputNpointsPerPillarDevice, pillar_num * sizeof(int)));

    // 2d vector复制时需要注意, 分配临时主机内存
    Point* tempHostMemoryPillar = new Point[pillar_num * max_points];
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
    CHECK(cudaMemcpy(inputPillarsDevice, tempHostMemoryPillar, pillar_num * max_points * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(inputCoorsBatchDevice, tempHostMemoryCoor, pillar_num * 4 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(inputNpointsPerPillarDevice, num_points_per_voxel.data(), pillar_num * sizeof(int), cudaMemcpyHostToDevice));
    delete[] tempHostMemoryPillar;
    delete[] tempHostMemoryCoor;
    // delete[] currentHostPtrPillar;
    // delete[] currentHostPtrCoor;

    // 分配输出设备内存
    void* outputDevice;
    CHECK(cudaMalloc(&outputDevice, output.size() * sizeof(float))); 

    // 设置输入张量的维度
    nvinfer1::Dims inputPilllarDims, inputCoorsDims, inputNpointsPerPillarDims; // 您期望的输入维度
    inputPilllarDims.nbDims = 3; // 维度数
    inputPilllarDims.d[0] = pillar_num; // 每个维度的大小
    inputPilllarDims.d[1] = max_points;
    inputPilllarDims.d[2] = sizeof(Point) / sizeof(float);

    inputCoorsDims.nbDims = 2; // 维度数
    inputCoorsDims.d[0] = pillar_num; // 每个维度的大小
    inputCoorsDims.d[1] = 4;

    inputNpointsPerPillarDims.nbDims = 1; // 维度数
    inputNpointsPerPillarDims.d[0] = pillar_num; // 每个维度的大小

    // 在推理之前设置输入张量的维度
    if (!context->setBindingDimensions(inputPillarsIndex, inputPilllarDims)) {
        // 处理错误，设置维度失败
        std::cout << "setBindingDimensions error \n";
    }
    if (!context->setBindingDimensions(inputCoorsBatchIndex, inputCoorsDims)) {
        std::cout << "setBindingDimensions error \n";
    }
    if (!context->setBindingDimensions(inputNpointsPerPillarIndex, inputNpointsPerPillarDims)) {
        std::cout << "setBindingDimensions error \n";
    }

    // 创建输入和输出数据缓冲区指针数组
    void* buffers[4];
    buffers[inputPillarsIndex] = inputPillarsDevice;
    buffers[inputCoorsBatchIndex] = inputCoorsBatchDevice;
    buffers[inputNpointsPerPillarIndex] = inputNpointsPerPillarDevice;
    buffers[outputIndex] = outputDevice;

    // 执行推理
    context->enqueueV2(buffers, 0, nullptr);

    // 如果需要，将输出数据从设备复制回主机
    cudaMemcpy(output.data(), outputDevice, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < output.size(); i++){
    //     if (i % 11 == 0) std::cout << (i / 11) << ": ";
    //     std::cout << output[i] << " ";
    //     if ((i + 1) % 11 == 0){
    //         std::cout << std::endl;
    //     }
    // }
    
    // 释放设备内存
    cudaFree(inputPillarsDevice);
    cudaFree(inputCoorsBatchDevice);
    cudaFree(inputNpointsPerPillarDevice);
    cudaFree(outputDevice);
}

void postProcessing(std::vector<float>& output, int& num_class, float& nms_thr, float& score_thr, 
                     int& max_num, std::vector<Box3dfull>& bboxes_full){
    std::vector<Box2d> bboxes_2d;
    std::vector<Box3d> bboxes_3d;
    std::vector<std::vector<float>> scores_list;
    std::vector<float> direction_list;
    decodeDetResults(output, num_class, bboxes_2d, bboxes_3d, scores_list, direction_list);

    std::vector<Box3dfull> bboxes_3d_nms;
    for (int i = 0; i < num_class; i++){
        std::vector<int> score_filter_inds;
        std::vector<float> scores;
        filterByScores(i, scores_list, score_thr, score_filter_inds, scores);
        std::vector<Box2d> bboxes_2d_filtered;
        std::vector<Box3d> bboxes_3d_filtered;
        std::vector<float> direction_filtered;
        obtainBoxByInds(score_filter_inds, bboxes_2d, bboxes_2d_filtered, bboxes_3d, bboxes_3d_filtered, 
                        direction_list, direction_filtered);
        
        std::vector<int> nms_filter_inds;
        nms(bboxes_2d_filtered, scores, nms_thr, nms_filter_inds);

        for (const auto ind : nms_filter_inds){
            Box3dfull box3d_full;
            box3d_full.x = bboxes_3d_filtered[ind].x;
            box3d_full.y = bboxes_3d_filtered[ind].y;
            box3d_full.z = bboxes_3d_filtered[ind].z;
            box3d_full.w = bboxes_3d_filtered[ind].w;
            box3d_full.l = bboxes_3d_filtered[ind].l;
            box3d_full.h = bboxes_3d_filtered[ind].h;
            float limited_theta = limitPeriod(bboxes_3d_filtered[ind].theta);
            box3d_full.theta = (1.f - direction_filtered[ind]) * M_PI + limited_theta;
            box3d_full.score = scores[ind];
            box3d_full.label = i;
            bboxes_3d_nms.push_back(box3d_full);
        }
    }
    getTopkBoxes(bboxes_3d_nms, max_num, bboxes_full);
}

void runTime(std::vector<Point>& points, int& test_number, nvinfer1::ICudaEngine* engine, 
             nvinfer1::IExecutionContext* context){
    std::vector<float> voxel_size = {0.16, 0.16, 4};
    std::vector<float> coors_range = {0, -39.68, -3, 69.12, 39.68, 1};
    int max_points = 32;
    int max_voxels = 40000;
    int num_class = 3, num_box = 100;
    float nms_thr = 0.01, score_thr = 0.1;
    int max_num = 50;
    
    std::chrono::duration<double, std::milli> voxelization_time(0);
    std::chrono::duration<double, std::milli> inference_time(0);
    std::chrono::duration<double, std::milli> post_processing_time(0);
    std::chrono::duration<double, std::milli> total_time(0);

    auto start_total = std::chrono::high_resolution_clock::now(); 
    for (int i = 0; i < test_number; i++){
        std::vector<Voxel> voxels;
        std::vector<std::vector<int>> coors;
        std::vector<int> num_points_per_voxel;

        // 1. voxelization
        auto start_voxelization = std::chrono::high_resolution_clock::now();
        voxelize(points, voxel_size, coors_range, max_points, max_voxels, voxels, coors, num_points_per_voxel);
        std::vector<std::vector<int>> padded_coors;
        padCoors(coors, padded_coors);
        auto end_voxelization = std::chrono::high_resolution_clock::now();
        voxelization_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_voxelization - start_voxelization);

        // 2. trt inference
        auto start_inference = std::chrono::high_resolution_clock::now();
        std::vector<float> output(num_box * (7 + num_class + 1));
        trtInfer(voxels, padded_coors, num_points_per_voxel, max_points, engine, context, output);
        auto end_inference = std::chrono::high_resolution_clock::now();
        inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
        
        // 3. post processing
        auto start_post_processing = std::chrono::high_resolution_clock::now();
        std::vector<Box3dfull> bboxes;
        postProcessing(output, num_class, nms_thr, score_thr, max_num, bboxes);
        auto end_post_processing = std::chrono::high_resolution_clock::now();
        post_processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_post_processing - start_post_processing);
    }
    auto end_total = std::chrono::high_resolution_clock::now(); 
    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

    double avg_voxelization_time = voxelization_time.count() / test_number;
    double avg_inference_time = inference_time.count() / test_number;
    double avg_post_processing_time = post_processing_time.count() / test_number;
    double avg_total_time = total_time.count() / test_number; 

    std::cout << "Average voxelization time: " << avg_voxelization_time << " ms" << std::endl;
    std::cout << "Average inference time: " << avg_inference_time << " ms" << std::endl;
    std::cout << "Average post processing time: " << avg_post_processing_time << " ms" << std::endl;
    std::cout << "Average total time for loop: " << avg_total_time << " ms" << std::endl; 
}

int main(int argc, char *argv[]) {
    if (argc != 3){
        std::cerr << "Usage: " << argv[0] << " your_point_cloud_path your_trt_path\n";
        return 1;
    }

    // 0. read data
    std::vector<Point> points_ori, points; 
    std::string file_path = argv[1]; // "../../../dataset/demo_data/val/000134.bin"
    bool read_data_ok = readPoints(file_path, points_ori);
    if (!read_data_ok) return 0;
    pointCloudFiler(points_ori, points);

    std::vector<float> voxel_size = {0.16, 0.16, 4};
    std::vector<float> coors_range = {0, -39.68, -3, 69.12, 39.68, 1};
    int max_points = 32;
    int max_voxels = 40000;
    
    std::vector<Voxel> voxels;
    std::vector<std::vector<int>> coors;
    std::vector<int> num_points_per_voxel;

    // 1. voxelization
    voxelize(points, voxel_size, coors_range, max_points, max_voxels, voxels, coors, num_points_per_voxel);
    std::vector<std::vector<int>> padded_coors;
    padCoors(coors, padded_coors);

    // 2. trt inference
    const std::string trt_path = argv[2];
    int num_class = 3, num_box = 100;
    std::vector<float> output(num_box * (7 + num_class + 1));
    auto components = initializeTensorRTComponents(trt_path);
    nvinfer1::ICudaEngine* engine = components.first;
    nvinfer1::IExecutionContext* context = components.second;
    trtInfer(voxels, padded_coors, num_points_per_voxel, max_points, engine, context, output);
    
    // 3. post processing
    float nms_thr = 0.01, score_thr = 0.1;
    int max_num = 50;
    std::vector<Box3dfull> bboxes;
    postProcessing(output, num_class, nms_thr, score_thr, max_num, bboxes);

    // 4. write results to file
    writeFile(bboxes, "../../infer_results/trt.txt");

    // 5. runtime 
    int test_number = 100;
    runTime(points, test_number, engine, context);

    context->destroy();
    engine->destroy();
    return 0;
}