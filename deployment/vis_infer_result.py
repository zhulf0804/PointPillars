import argparse
import numpy as np
import os
import sys

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR))

from utils import read_points, vis_pc


def read_det_result(file_path):
    lidar_bboxes, labels = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = list(line.strip().split())
            lidar_bboxes.append(list(map(float, items[:7])))
            labels.append(int(float(items[-1])))
    return np.array(lidar_bboxes), np.array(labels, dtype=np.int32)


def vis_result(args):
    pc = read_points(args.pc_path)
    lidar_bboxes, labels = read_det_result(args.result_path)
    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--pc_path', required=True, help='your point cloud path')
    parser.add_argument('--result_path', required=True,
                        help='your saved path for detection results')
    args = parser.parse_args()
    vis_result(args)

# python vis_infer_result.py --pc_path ../dataset/demo_data/val/000134.bin  --result_path infer_results/pytorch.txt
# python vis_infer_result.py --pc_path ../dataset/demo_data/val/000134.bin  --result_path infer_results/onnx.txt
# python vis_infer_result.py --pc_path ../dataset/demo_data/val/000134.bin  --result_path infer_results/trt.txt
# python vis_infer_result.py --pc_path ../dataset/demo_data/val/000134.bin  --result_path infer_results/trt-fp16.txt