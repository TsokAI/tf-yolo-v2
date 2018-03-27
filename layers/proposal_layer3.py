from __future__ import absolute_import, division, print_function
import numpy as np
from utils.bbox_transform import bbox_transform_inv3
import config as cfg


def proposal_layer(bbox_pred, iou_pred, cls_pred, anchors, logitsize):
    box_pred = bbox_transform_inv3(
        np.ascontiguousarray(bbox_pred, dtype=np.float32),
        np.ascontiguousarray(anchors, dtype=np.float32),
        cfg.INP_SIZE, cfg.INP_SIZE,
        logitsize, logitsize)

    box_pred = np.reshape(box_pred, [-1, 4])

    iou_pred = np.reshape(iou_pred, [-1, 1])

    cls_pred = np.reshape(cls_pred, [-1, cfg.NUM_CLASSES])

    cls_inds = np.argmax(cls_pred, axis=1)
    cls_prob = cls_pred[np.arange(cls_pred.shape[0]), cls_inds][:, np.newaxis]

    scores = iou_pred * cls_prob

    # filter out boxes with scores <= coef thresh
    keep = np.where(scores >= cfg.COEF_THRESH)[0]
    # keep top n scores before apply nms
    keep = keep[np.argsort(-scores[keep, 0])[:cfg.PRE_NMS_TOP_N]]

    box_pred = box_pred[keep]
    cls_inds = cls_inds[keep]
    scores = scores[keep]

    return box_pred, cls_inds, scores
