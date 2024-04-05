// common_trt.h
#ifndef COMMON_TRT_H
#define COMMON_TRT_H


#include <vector>

struct Point {
    float x, y, z, feature;
};

struct Voxel {
    std::vector<Point> points;
};

#endif // COMMON_TRT_H