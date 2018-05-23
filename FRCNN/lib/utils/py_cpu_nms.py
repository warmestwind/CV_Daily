# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
# 将scores从大到小排存在order里，递归的选择scores排在第一个的box即order[0]并存在keep里,
# 计算其与后面各个box的重叠率，选择小于0.7的box，更新order[];
# 重复以上过程直到order为空
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # xx/yy 交集对角坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #重叠率
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #only index 
        inds = np.where(ovr <= thresh)[0] # np.where return (index,type)

        # inds+1 because keep.append(order[0])
        order = order[inds + 1]

    return keep
