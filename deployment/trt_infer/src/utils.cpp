#include <algorithm>
#include "utils.h"

void decodeDetResults(std::vector<float>& output, int& num_class, std::vector<Box2d>& bboxes_2d,
                        std::vector<Box3d>& bboxes_3d, std::vector<std::vector<float>>& scores_list,
                        std::vector<float>& direction_list){
    int n_type = 7 + num_class + 1;
    for (int i = 0; i < output.size(); i+=n_type){
        Box3d box3d; 
        Box2d box2d;
        std::vector<float> scores;
        
        float x = output[i], y = output[i+1];
        float w = output[i+3], l = output[i+4];
        float angle = output[i+6];
        box2d.x1 = x - w / 2;
        box2d.y1 = y - l / 2;
        box2d.x2 = x + w / 2;
        box2d.y2 = y + l / 2;
        box2d.theta = angle;
        bboxes_2d.push_back(box2d);

        box3d.x = x;
        box3d.y = y;
        box3d.z = output[i+2];
        box3d.w = output[i+3];
        box3d.l = output[i+4];
        box3d.h = output[i+5];
        box3d.theta = angle;
        bboxes_3d.push_back(box3d);

        for(int j = 0; j < num_class; j++)
            scores.push_back(output[i+7+j]);
        scores_list.push_back(scores);

        direction_list.push_back(output[i+7+num_class]);
    }
}

void filterByScores(int& i, std::vector<std::vector<float>>& scores_list, float& score_thr, 
                    std::vector<int>& inds, std::vector<float>& scores){
    for (int j = 0; j < scores_list.size(); j++){
        if (scores_list[j][i] > score_thr){
            inds.push_back(j);
            scores.push_back(scores_list[j][i]);
        }
    }
}

void obtainBoxByInds(std::vector<int>& inds, std::vector<Box2d>& bboxes_2d, 
                     std::vector<Box2d>& bboxes_2d_filtered, std::vector<Box3d>& bboxes_3d, 
                     std::vector<Box3d>& bboxes_3d_filtered, std::vector<float>& direction_list,
                     std::vector<float>& direction_filtered){
    for (const auto ind : inds){
        bboxes_2d_filtered.push_back(bboxes_2d[ind]);
        direction_filtered.push_back(direction_list[ind]);
        bboxes_3d_filtered.push_back(bboxes_3d[ind]);
    }
}

float limitPeriod(float& val, float offset, float period){
    float limited_val = val - std::floor(val / period + offset) * period;
    return limited_val;
}

bool compareByScore(const Box3dfull& a, const Box3dfull& b) {
    return a.score > b.score;
}

void getTopkBoxes(std::vector<Box3dfull>& bboxes_3d_nms, int& max_num, std::vector<Box3dfull>& bboxes_full){
    bboxes_full.clear();
    if (bboxes_3d_nms.size() <= max_num){
        std::copy(bboxes_3d_nms.begin(), bboxes_3d_nms.end(), std::back_inserter(bboxes_full));
        return;
    }
    std::partial_sort(bboxes_3d_nms.begin(), bboxes_3d_nms.begin() + max_num, bboxes_3d_nms.end(), compareByScore);
    std::copy(bboxes_3d_nms.begin(), bboxes_3d_nms.begin() + max_num, std::back_inserter(bboxes_full));
}