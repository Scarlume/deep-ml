import numpy as np
import math

def get_iou(box1, box2):
    """
    计算两个矩形框的 IoU。

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        float: IoU，范围 [0, 1]
    """

    # 1. 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 2. 计算交集面积
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # 3. 计算两个矩形框各自的面积
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    # 4. 计算并集面积
    union_area = area1 + area2 - inter_area

    # 5. 计算 IoU
    if union_area == 0:
        return 0.0

    return inter_area / union_area
     

def soft_nms(boxes, score, method="gaussian", iou_threshold=0.3, sigma=0.5, score_threshold=0.001):
    score=list(score)
    indexes = [i for i in range(len(boxes))]
    res = []
    while len(indexes)>0:
        max_score_index = max(indexes,key = lambda x:(score[x],-x))
        res.append((max_score_index,float(score[max_score_index])))
        indexes.remove(max_score_index)
        
        for idx in indexes:
            cur_index = idx
            iou = get_iou(boxes[max_score_index], boxes[cur_index])
            if method == "linear":
                if iou > iou_threshold:
                    score[cur_index] *= (1-iou)
            elif method == "hard":
                if iou > iou_threshold:
                    score[cur_index] = 0
            elif method == "gaussian":
                score[cur_index] *= math.exp(-(iou**2)/sigma)
        indexes = [idx for idx in indexes if score[idx]>=score_threshold]

    return res

            

