#ifndef VOXELIZATION_TRT_H
#define VOXELIZATION_TRT_H
#include <vector>

#include "common_trt.h"

void voxelize(const std::vector<Point>& points, const std::vector<float>& voxel_size, 
              const std::vector<float>& coors_range, int max_points, int max_voxels, 
              std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors,
              std::vector<int>& num_points_per_voxel);

void voxelizeGpu(const std::vector<Point>& points, const std::vector<float>& voxel_size,
                 const std::vector<float>& coors_range, int max_points, int max_voxels, 
                 std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors, 
                 std::vector<int>& num_points_per_voxel, const int NDim = 3);

int voxelizeGpu(const std::vector<Point>& points, const std::vector<float>& voxel_size,
                 const std::vector<float>& coors_range, int max_points, int max_voxels, 
                 float* d_voxels, int* d_coors, int* d_num_points_per_voxel, const int NDim);

void padCoors(std::vector<std::vector<int>>& coors, std::vector<std::vector<int>>& padded_coors);

void padCoorsGPU(int* d_coors, int* d_coors_padded, int& voxel_num);
#endif // VOXELIZATION_TRT_H