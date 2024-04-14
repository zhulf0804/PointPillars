#include "voxelization_trt.h"

void voxelize(const std::vector<Point>& points, const std::vector<float>& voxel_size, 
              const std::vector<float>& coors_range, int max_points, int max_voxels, 
              std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors,
              std::vector<int>& num_points_per_voxel) {
    
    // 计算网格尺寸
    int grid_size_x = static_cast<int>((coors_range[3] - coors_range[0]) / voxel_size[0]);
    int grid_size_y = static_cast<int>((coors_range[4] - coors_range[1]) / voxel_size[1]);
    int grid_size_z = static_cast<int>((coors_range[5] - coors_range[2]) / voxel_size[2]);
    
    for (const auto& point : points) {
        // 计算体素坐标
        int x = static_cast<int>((point.x - coors_range[0]) / voxel_size[0]);
        int y = static_cast<int>((point.y - coors_range[1]) / voxel_size[1]);
        int z = static_cast<int>((point.z - coors_range[2]) / voxel_size[2]);
        
        // 检查点是否在网格范围内
        if (x < 0 || x >= grid_size_x || y < 0 || y >= grid_size_y || z < 0 || z >= grid_size_z) {
            continue;
        }
        
        // 查找体素
        int k = -1;
        for (int i = 0; i < coors.size(); i++) {
            if (coors[i][0] == x && coors[i][1] == y && coors[i][2] == z) {
                k = i;
                break;
            }
        }
        
        if (k == -1 && voxels.size() < max_voxels) {
            // 创建新的体素
            Voxel voxel;
            voxel.points.resize(max_points);        
            voxels.push_back(voxel);
            coors.push_back({x, y, z});
            num_points_per_voxel.push_back(1);
            
            Voxel &voxel_back = voxels.back();
            voxel_back.points[0].x = point.x;
            voxel_back.points[0].y = point.y;
            voxel_back.points[0].z = point.z;
            voxel_back.points[0].feature = point.feature;
        }
        
        if (k != -1 && num_points_per_voxel[k] < max_points) {
            // 添加点到体素中
            voxels[k].points[num_points_per_voxel[k]].x = point.x;
            voxels[k].points[num_points_per_voxel[k]].y = point.y;
            voxels[k].points[num_points_per_voxel[k]].z = point.z;
            voxels[k].points[num_points_per_voxel[k]].feature = point.feature;
            num_points_per_voxel[k] += 1;
        }
    }
    
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

    return;
}

void padCoors(std::vector<std::vector<int>>& coors, std::vector<std::vector<int>>& padded_coors){
    for (auto& coor : coors){
        std::vector<int> padded_coor = {0};
        padded_coor.insert(padded_coor.end(), coor.begin(), coor.end());
        padded_coors.push_back(padded_coor);
    }
}