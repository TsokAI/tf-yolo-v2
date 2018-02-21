from __future__ import absolute_import, division, print_function
import numpy as np
from utils.bbox import box_overlaps
from config import NUM_CLASSES


def compute_ap(recall, precision):
    # eval/voc_ap in https://github.com/rbgirshick/py-faster-rcnn

    # append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # calculate AUC, points recall changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_image(box_pred, cls_inds, gt_boxes, gt_cls):
    pass
