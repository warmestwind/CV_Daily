# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
#解读
#https://blog.csdn.net/qq_32473685/article/details/80058768

# 根据anchors和rpn_bbox_pred回归候选框并提取有物体框的分数，然后提取RPN_PRE_NMS_TOP_N个，
# 接着NMS，最后再提取RPN_POST_NMS_TOP_N个，返回变量为rois， rpn_scores。
#[array([ 1, 50, 38, 18]), array([ 1, 50, 38, 36])]                             [none,4]
def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n #12000
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n #2000
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh #0.7
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    im_info = im_info[0] #[batch,3]
    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]  #[ 1, 50, 38, 18] =》[1, 50, 38, 9:18]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4)) # [ 1, 50, 38, 36] =》[-1, 4],把最后一维每四个一行
    scores = scores.reshape((-1, 1)) #[17100，1]
    #返回经过反变换的bbox
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred) #[none,4] [17100,4],锚点，预测值,
    proposals = clip_boxes(proposals, im_info[:2]) #将bbox坐标剪切到原图内

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1] #argsort() 返回从小到大索引值, order[]存scores从大到小索引
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :] #[12000,4] 
    scores = scores[order] #[12000,1]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN] #2000, keep save index
    proposals = proposals[keep, :] #(2000,4)
    scores = scores[keep] #(2000,1)

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    # (2000, 1+4) (2000, 1)
    return blob, scores
