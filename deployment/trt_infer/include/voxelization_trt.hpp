#include "common_trt.h"
#include <vector>


void voxelize(const std::vector<Point>& points, const std::vector<float>& voxel_size, 
              const std::vector<float>& coors_range, int max_points, int max_voxels, 
              std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors,
              std::vector<int>& num_points_per_voxel);

void pad_coors(std::vector<std::vector<int>>& coors, std::vector<std::vector<int>>& padded_coors);