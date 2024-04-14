#ifndef UTILS_H
#define UTILS_H
#include<cmath>

#include "common_trt.h"

void decodeDetResults(std::vector<float>& output, int& num_class, std::vector<Box2d>& bboxes_2d,
                      std::vector<Box3d>& bboxes_3d, std::vector<std::vector<float>>& scores_list,
                      std::vector<float>& direction_list);

void filterByScores(int& i, std::vector<std::vector<float>>& scores_list, float& score_thr, 
                    std::vector<int>& inds, std::vector<float>& scores);

void obtainBoxByInds(std::vector<int>& inds, std::vector<Box2d>& bboxes_2d, 
                     std::vector<Box2d>& bboxes_2d_filtered, std::vector<Box3d>& bboxes_3d, 
                     std::vector<Box3d>& bboxes_3d_filtered, std::vector<float>& direction_list,
                     std::vector<float>& direction_filtered);

float limitPeriod(float& val, float offset=1.f, float period=M_PI);

bool compareByScore(const Box3dfull& a, const Box3dfull& b);

void getTopkBoxes(std::vector<Box3dfull>& bboxes_3d_nms, int& max_num, std::vector<Box3dfull>& bboxes_full);
#endif // UTILS_H