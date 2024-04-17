#include <cuda_runtime.h>
#include "voxelization_trt.h"

// Modified from https://github.com/zhulf0804/PointPillars/blob/main/ops/voxelization/voxelization_cuda.cu 
// Modified from mmdet3d

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          return;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    // if (coor[i][0] == -1)
    //    continue;
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = NDim - 1 - thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

__global__ void pad(int* d_coors, int* d_coors_padded, int voxel_num){
    CUDA_1D_KERNEL_LOOP(index, voxel_num) {
        auto coor_offset = d_coors + index * 3;
        auto coor_offset_padded = d_coors_padded + index * 4;
        coor_offset_padded[1] = coor_offset[0];
        coor_offset_padded[2] = coor_offset[1];
        coor_offset_padded[3] = coor_offset[2];
    }
}

// Copy result on device to host
void voxelizeGpu(const std::vector<Point>& points, const std::vector<float>& voxel_size,
                  const std::vector<float>& coors_range, int max_points, int max_voxels, 
                  std::vector<Voxel>& voxels, std::vector<std::vector<int>>& coors, 
                  std::vector<int>& num_points_per_voxel, const int NDim) {
    const int num_points = points.size();
    const int num_features = sizeof(Point) / sizeof(float);

    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    float* d_points;
    cudaMalloc(&d_points, points.size() * sizeof(Point));
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
    int *temp_coors; 
    cudaMalloc((void **)&temp_coors, num_points * NDim * sizeof(int));

    dim3 grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 block(512);

    // 1. link point to corresponding voxel coors
    dynamic_voxelize_kernel<float, int>
        <<<grid, block>>>(
            d_points, temp_coors, voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min, 
            coors_z_min, coors_x_max, coors_y_max, coors_z_max, grid_x, grid_y, grid_z, 
            num_points, num_features, NDim);
    cudaDeviceSynchronize();

    // 2. map point to the idx of the corresponding voxel, find duplicate coor
    // create some temporary variables    
    int* point_to_pointidx = nullptr; 
    cudaMalloc((void**)&point_to_pointidx, num_points * sizeof(int));
    cudaMemset(point_to_pointidx, -1, num_points * sizeof(int));
    int* point_to_voxelidx = nullptr; 
    cudaMalloc((void**)&point_to_voxelidx, num_points * sizeof(int));
    cudaMemset(point_to_voxelidx, -1, num_points * sizeof(int));

    dim3 map_grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 map_block(512);
    point_to_voxelidx_kernel<int>
        <<<map_grid, map_block>>>(
            temp_coors, point_to_voxelidx, point_to_pointidx, max_points, max_voxels,
            num_points, NDim);     
    cudaDeviceSynchronize();

    // 3. determin voxel num and voxel's coor index
    // make the logic in the CUDA device could accelerate about 10 times
    int* coor_to_voxelidx = nullptr; 
    cudaMalloc((void**)&coor_to_voxelidx, num_points * sizeof(int));
    cudaMemset(coor_to_voxelidx, -1, num_points * sizeof(int));
    int* voxel_num = nullptr; 
    cudaMalloc((void**)&voxel_num, 1 * sizeof(int));
    cudaMemset(voxel_num, 0, 1 * sizeof(int));
    int* d_num_points_per_voxel = nullptr;
    cudaMalloc((void**)&d_num_points_per_voxel, max_voxels * sizeof(int));
    cudaMemset(d_num_points_per_voxel, 0, max_voxels * sizeof(int));

    determin_voxel_num<int><<<1, 1>>>(
        d_num_points_per_voxel, point_to_voxelidx, point_to_pointidx, coor_to_voxelidx,
        voxel_num, max_points, max_voxels, num_points);
    cudaDeviceSynchronize();

    // 4. copy point features to voxels
    // Step 4 & 5 could be parallel
    auto pts_output_size = num_points * num_features;
    dim3 cp_grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 cp_block(512);
    float* d_voxels = nullptr;
    cudaMalloc((void**)&d_voxels, max_voxels * max_points * sizeof(Point));
    cudaMemset(d_voxels, 0.f, max_voxels * max_points * sizeof(Point));

    assign_point_to_voxel<float, int>
        <<<cp_grid, cp_block>>>(
            pts_output_size, d_points, point_to_voxelidx, coor_to_voxelidx,
            d_voxels, max_points, num_features,
            num_points, NDim);
    //   cudaDeviceSynchronize();

    // 5. copy coors of each voxels
    auto coors_output_size = num_points * NDim;
    dim3 coors_cp_grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 coors_cp_block(512);
    int* d_coors = nullptr;
    cudaMalloc((void**)&d_coors, max_voxels * NDim * sizeof(int));
    cudaMemset(d_coors, 0, max_voxels * NDim * sizeof(int));

    assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block>>>(
        coors_output_size, temp_coors,
        point_to_voxelidx,
        coor_to_voxelidx,
        d_coors, num_points, NDim);
    cudaDeviceSynchronize();

    // 6. copy device data to host
    int* cpu_voxel_num = new int[1];
    cudaMemcpy(cpu_voxel_num, voxel_num, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    int voxel_num_val = *cpu_voxel_num;

    voxels.resize(voxel_num_val);
    size_t voxel_offset = 0;
    for (size_t i = 0; i < voxel_num_val; ++i) {
        size_t voxel_size = max_points * sizeof(Point);
        voxels[i].points.resize(max_points);
        cudaMemcpy(voxels[i].points.data(), d_voxels + voxel_offset, voxel_size, cudaMemcpyDeviceToHost);
        voxel_offset += max_points * sizeof(Point) / sizeof(float);
    }

    coors.resize(voxel_num_val);
    size_t coor_offset = 0;
    for (size_t i = 0; i < voxel_num_val; ++i) {
        size_t coor_size = NDim * sizeof(int);
        coors[i].resize(NDim);
        cudaMemcpy(coors[i].data(), d_coors + coor_offset, coor_size, cudaMemcpyDeviceToHost);
        coor_offset += NDim;
    }
    
    num_points_per_voxel.resize(voxel_num_val);
    cudaMemcpy(num_points_per_voxel.data(), d_num_points_per_voxel, voxel_num_val * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 7. free memory
    cudaFree(d_points);
    cudaFree(temp_coors);
    cudaFree(point_to_pointidx);
    cudaFree(point_to_voxelidx);
    cudaFree(coor_to_voxelidx);
    cudaFree(voxel_num);
    cudaFree(d_num_points_per_voxel);
    cudaFree(d_voxels);
    cudaFree(d_coors);

    delete[] cpu_voxel_num;
}

// Keep results on device for faster speed
int voxelizeGpu(const std::vector<Point>& points, const std::vector<float>& voxel_size,
                 const std::vector<float>& coors_range, int max_points, int max_voxels, 
                 float* d_voxels, int* d_coors, int* d_num_points_per_voxel, const int NDim) {
    const int num_points = points.size();
    const int num_features = sizeof(Point) / sizeof(float);

    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    float* d_points;
    cudaMalloc(&d_points, points.size() * sizeof(Point));
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
    int *temp_coors; 
    cudaMalloc((void **)&temp_coors, num_points * NDim * sizeof(int));

    dim3 grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 block(512);

    // 1. link point to corresponding voxel coors
    dynamic_voxelize_kernel<float, int>
        <<<grid, block>>>(
            d_points, temp_coors, voxel_x, voxel_y, voxel_z, coors_x_min, coors_y_min, 
            coors_z_min, coors_x_max, coors_y_max, coors_z_max, grid_x, grid_y, grid_z, 
            num_points, num_features, NDim);
    cudaDeviceSynchronize();

    // 2. map point to the idx of the corresponding voxel, find duplicate coor
    // create some temporary variables    
    int* point_to_pointidx = nullptr; 
    cudaMalloc((void**)&point_to_pointidx, num_points * sizeof(int));
    cudaMemset(point_to_pointidx, -1, num_points * sizeof(int));
    int* point_to_voxelidx = nullptr; 
    cudaMalloc((void**)&point_to_voxelidx, num_points * sizeof(int));
    cudaMemset(point_to_voxelidx, -1, num_points * sizeof(int));

    dim3 map_grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 map_block(512);
    point_to_voxelidx_kernel<int>
        <<<map_grid, map_block>>>(
            temp_coors, point_to_voxelidx, point_to_pointidx, max_points, max_voxels,
            num_points, NDim);     
    cudaDeviceSynchronize();

    // 3. determin voxel num and voxel's coor index
    // make the logic in the CUDA device could accelerate about 10 times
    int* coor_to_voxelidx = nullptr; 
    cudaMalloc((void**)&coor_to_voxelidx, num_points * sizeof(int));
    cudaMemset(coor_to_voxelidx, -1, num_points * sizeof(int));
    int* voxel_num = nullptr; 
    cudaMalloc((void**)&voxel_num, 1 * sizeof(int));
    cudaMemset(voxel_num, 0, 1 * sizeof(int));

    determin_voxel_num<int><<<1, 1>>>(
        d_num_points_per_voxel, point_to_voxelidx, point_to_pointidx, coor_to_voxelidx,
        voxel_num, max_points, max_voxels, num_points);
    cudaDeviceSynchronize();

    // 4. copy point features to voxels
    // Step 4 & 5 could be parallel
    auto pts_output_size = num_points * num_features;
    dim3 cp_grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 cp_block(512);
    
    assign_point_to_voxel<float, int>
        <<<cp_grid, cp_block>>>(
            pts_output_size, d_points, point_to_voxelidx, coor_to_voxelidx,
            d_voxels, max_points, num_features,
            num_points, NDim);
    //   cudaDeviceSynchronize();

    // 5. copy coors of each voxels
    auto coors_output_size = num_points * NDim;
    dim3 coors_cp_grid(std::min(static_cast<int>(std::ceil(num_points / 512)), 4096));
    dim3 coors_cp_block(512);

    assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block>>>(
        coors_output_size, temp_coors,
        point_to_voxelidx,
        coor_to_voxelidx,
        d_coors, num_points, NDim);
    cudaDeviceSynchronize();

    // 6. copy device data to host
    int* cpu_voxel_num = new int[1];
    cudaMemcpy(cpu_voxel_num, voxel_num, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    int voxel_num_val = *cpu_voxel_num;

    // 7. free memory
    cudaFree(d_points);
    cudaFree(temp_coors);
    cudaFree(point_to_pointidx);
    cudaFree(point_to_voxelidx);
    cudaFree(coor_to_voxelidx);
    
    delete[] cpu_voxel_num;

    return voxel_num_val;
}

void padCoorsGPU(int* d_coors, int* d_coors_padded, int& voxel_num){
    dim3 grid(std::min(static_cast<int>(std::ceil(voxel_num / 512)), 4096));
    dim3 block(512);
    pad<<<grid, block>>>(d_coors, d_coors_padded, voxel_num);
}