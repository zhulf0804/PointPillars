#ifndef IO_TRT_HPP
#define IO_TRT_HPP
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "common_trt.h"

void pointCloudFiler(std::vector<Point>& points, std::vector<Point>& new_points){
    const float data_range[] = {0, -39.68, -3, 69.12, 39.68, 1};
    for (const auto& point : points){
        if (point.x > data_range[0] && point.x < data_range[3]
            && point.y > data_range[1] && point.y < data_range[4]
            && point.z > data_range[2] && point.z < data_range[5]){
                new_points.emplace_back(point);
            }
    }

    // // 输出数据（测试用）
    // for (auto point : new_points) {
    //     std::cout << point.x << ", " << point.y << ", " << point.z << " " << point.feature << std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << "points size: " << new_points.size() << std::endl;
}

bool readPoints(std::string file_path, std::vector<Point>& points){
    // 打开文件
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 0;
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 分配内存
    std::vector<char> buffer(size);

    // 读取数据
    file.read(buffer.data(), size);

    // 转换数据类型（如果需要）
    std::vector<float> data(size / sizeof(float));
    std::memcpy(data.data(), buffer.data(), size);

    // 关闭文件
    file.close();

    points.reserve(data.size() / 4); // 每个Point由4个float组成

    for (size_t i = 0; i < data.size(); i += 4) {
        Point p = {data[i], data[i + 1], data[i + 2], data[i + 3]};
        points.push_back(p);
    }

    return 1;
}

// 读取引擎文件
std::vector<char> readEngineFile(const std::string& enginePath)
{
    // 使用 std::ifstream 打开一个文件流，以二进制模式（std::ios::binary）和读取位置指向文件末尾（std::ios::ate）的方式打开文件。
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    // 通过 file.good() 检查文件流状态是否正常，如果不正常（即文件打开失败或其他错误），则抛出一个 std::runtime_error 异常。
    if (!file.good())
    {
        throw std::runtime_error("Error reading engine file");
    }

    // 使用 file.tellg() 获取当前文件指针位置，由于文件是以 std::ios::ate 模式打开的，所以这将给出文件的大小。
    std::streamsize size = file.tellg();

    // 通过 file.seekg(0, std::ios::beg) 将文件指针重新定位到文件的开始位置。
    file.seekg(0, std::ios::beg);

    // 创建一个足够大的 std::vector<char> 来存储文件内容，其大小由步骤3中获取的文件大小确定。
    std::vector<char> buffer(size);

    // 使用 file.read(buffer.data(), size) 将文件内容读取到之前创建的向量中。buffer.data() 提供了向量内存的指针，size 是要读取的字节数。
    // 如果读取失败（即 file.read() 返回 false），则抛出一个 std::runtime_error 异常。
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Error reading engine file");
    }

    return buffer;
}

void writeFile(std::vector<Box3dfull>& bboxes, const std::string file_path){
    std::ofstream outfile(file_path);

    // 检查文件是否成功打开
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file " << file_path << " for writing." << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(4); 
    // 遍历 bboxes 中的每个元素，并写入文件
    for (const auto& bbox_3d : bboxes) {
        outfile << bbox_3d.x << " " << bbox_3d.y << " " << bbox_3d.z << " "
                << bbox_3d.w << " " << bbox_3d.l << " " << bbox_3d.h << " "
                << bbox_3d.theta << " " << bbox_3d.score << " " << bbox_3d.label << std::endl;
    }

    // 关闭文件
    outfile.close();
}
#endif // IO_TRT_HPP