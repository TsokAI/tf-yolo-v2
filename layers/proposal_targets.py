from __future__ import absolute_import, division, print_function
import numpy as np
from numba import jit, prange
from utils.bbox import box_overlaps, anchor_overlaps
import config as cfg


def compute_targets(box_pred, iou_pred, gt_boxes, gt_cls, ft_shape, anchors):
    # remove padding boxes, cls from groundtruth
    keep = np.where(gt_cls >= 0)[0]
    gt_boxes = gt_boxes[keep]
    gt_cls = gt_cls[keep]


@jit(nopython=True, nogil=True, cache=True, parallel=True)
def proposal_targets():
    pass
