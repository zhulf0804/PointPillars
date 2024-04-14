#include <numeric>
#include <algorithm>

#include "nms_trt.h"

float iou(const Box2d& a, const Box2d& b) {
    float interArea = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                      std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float unionArea = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - interArea;
    return interArea / unionArea;
}

void nms(std::vector<Box2d>& bboxes_2d_filtered, std::vector<float>& scores, 
         float& nms_thr, std::vector<int>& nms_filter_inds) {
    std::vector<int> indices(bboxes_2d_filtered.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });

    while (!indices.empty()) {
        int idx = indices.front();
        nms_filter_inds.push_back(idx);
        
        indices.erase(indices.begin());
        
        for (auto it = indices.begin(); it != indices.end(); ) {
            if (iou(bboxes_2d_filtered[idx], bboxes_2d_filtered[*it]) > nms_thr) {
                it = indices.erase(it);
            } else {
                ++it;
            }
        }
    }
}