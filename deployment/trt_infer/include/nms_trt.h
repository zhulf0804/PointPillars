#ifndef NMS_TRT_H
#define NMS_TRT_H

#include "common_trt.h"

float iou(const Box2d& a, const Box2d& b);

void nms(std::vector<Box2d>& bboxes_2d_filtered, std::vector<float>& scores, 
         float& nms_thr, std::vector<int>& nms_filter_inds);
#endif // NMS_TRT_H