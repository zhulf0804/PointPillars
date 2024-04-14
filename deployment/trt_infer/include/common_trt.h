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

// used for iou calculation
struct Box2d {
    float x1, y1, x2, y2, theta;
};

struct Box3d {
    float x, y, z, w, l, h, theta;
};

struct Box3dfull {
    float x, y, z, w, l, h, theta, score;
    int label;
};
#endif // COMMON_TRT_H