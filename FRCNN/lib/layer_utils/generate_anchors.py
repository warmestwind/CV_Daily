
import numpy as np
def generate_anchors_pre(height=38+1, width=50+1, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0] # # shape 9*4
    shift_x = np.arange(0, width) * feat_stride #[) 50 position
    shift_y = np.arange(0, height) * feat_stride #38
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) #网格化  38,50   38 50  (Ny, Nx)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0] # 1900,4
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)) #1900, 9,4, broadcast
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False) #1900*9,4
    length = np.int32(anchors.shape[0]) #17100

    return anchors, length

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios) #3 ratio 
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) #3 scale every ratio
                         for i in range(ratio_anchors.shape[0])])
    return anchors

def _ratio_enum(anchor, ratios): #列举关于一个anchor的三种宽高比 1:2,1:1,2:1
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    # fix size , w*w*r=size ,w = sqrt(size/r )
    w, h, x_ctr, y_ctr = _whctrs(anchor) # w:16,h:16,x_ctr:7.5,y_ctr:7.5 
    size = w * h #256  #面积一定
    size_ratios = size / ratios    
    ws = np.round(np.sqrt(size_ratios)) # 23 16 11  raster only can be int
    hs = np.round(ws * ratios) # 12 16 22，  sqrt(size/ratio) *  sqrt(size/ratio) *ratio = size
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales): #列举关于一个anchor的三种尺度 128*128,256*256,512*512 
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales #放大
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)#得到新锚点
    return anchors

def _whctrs(anchor): #anchor 存的是对角坐标，函数返回中心点，宽高
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr): #根据宽高，中心点坐标，计算对角坐标，即anchor
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis] # add 1-axis [[23],[16],[11]] 3x1
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

anchors, length = generate_anchors_pre()
print(anchors, length)
