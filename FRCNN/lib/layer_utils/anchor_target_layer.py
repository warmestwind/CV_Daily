# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform



def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors #9
    total_anchors = all_anchors.shape[0]  #17100
    K = total_anchors / num_anchors # 50*38
    im_info = im_info[0] # 800 600 1

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # 1, 50, 38, 18

    # only keep anchors inside the image, x1,y1 >0 , x2<width, y2<height
    inds_inside = np.where( 
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes（训练集的roi)
    # overlaps (ex, gt)
    # 计算所有gt[K,4]与所有inside_anchor[N,4]的overlap[N,K],
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float), # c-order
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1) # every row max index,每个anchor对应最大overlap的gt索引(N,)
    gt_argmax_overlaps = overlaps.argmax(axis=0)  #每个gt对应最大overlap的anchor索引(K)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, # K X K
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0] #max anchor index 

    if not cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives            
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0 #0.3

    # fg label: for each gt, anchor with highest overlap, 对于每个ground truth,与其最大IOU的anchor设置1
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU ，对于每个anchor,与gt的IOU大于0.7的设置为1
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1 

    if cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # subsample positive labels if we have too many
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize) #0.5*256，一半正例
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1 #随机选出多余128的索引，将label设为-1

    # subsample negative labels if we have too many
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1) #正例可能少于一半（128），所以负例来补 256-正例数
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False) #1D array, size
        labels[disable_inds] = -1

    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :]) #公式2，gt向anchor做归一化
    # pi* (inside weigths)
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS2["bbox_inside_weights"])#[1,1,1,1]

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.FLAGS.rpn_positive_weight < 0:  #-1
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples #正负例比例1:1，故权重相同
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.FLAGS.rpn_positive_weight > 0) &
                (cfg.FLAGS.rpn_positive_weight < 1))
        positive_weights = (cfg.FLAGS.rpn_positive_weight /  #p/n_pos
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.FLAGS.rpn_positive_weight) / #1-p/n_neg
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    # (1,1,9*50,38)  (1,50,38,36)

def _unmap(data, count, inds, fill=0): #恢复到17100个anchor对应的各种权重/标签
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data #二维 
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
