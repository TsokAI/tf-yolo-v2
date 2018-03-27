from __future__ import absolute_import, division, print_function
import numpy as np
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from layers.nms_wrapper import nms_detection
import config as cfg


def proposal_layer(bbox_pred, iou_pred, cls_pred, anchors, logitsize):
    box_coords = bbox_transform_inv(np.ascontiguousarray(bbox_pred, dtype=np.float32), np.ascontiguousarray(
        anchors, dtype=np.float32), logitsize, logitsize) * cfg.INP_SIZE

    box_coords = np.reshape(box_coords, [-1, 4])

    iou_pred = np.reshape(iou_pred, [-1, 1])

    cls_pred = np.reshape(cls_pred, [-1, cfg.NUM_CLASSES])
    box_cls = np.argmax(cls_pred, axis=1)

    cls_prob = cls_pred[np.arange(cls_pred.shape[0]), box_cls][:, np.newaxis]
    box_scores = iou_pred * cls_prob

    # filter out boxes with scores <= coef thresh
    keep = np.where(box_scores >= cfg.COEF_THRESH)[0]
    # keep top n scores before apply nms
    keep = keep[np.argsort(-box_scores[keep, 0])[:cfg.PRE_NMS_TOP_N]]

    box_coords = box_coords[keep]
    box_cls = box_cls[keep]
    box_scores = box_scores[keep]

    # apply nms with top-n-score boxes
    keep = np.zeros(len(box_coords), dtype=np.int8)
    for i in range(cfg.NUM_CLASSES):
        inds = np.where(box_cls == i)[0]
        if len(inds) == 0:
            continue

        dets = np.hstack([box_coords[inds], box_scores[inds]])

        keep_in_cls = nms_detection(np.ascontiguousarray(dets, dtype=np.float32),
                                    cfg.NMS_THRESH, use_gpu=cfg.USE_GPU)

        keep[inds[keep_in_cls]] = 1

    keep = np.where(keep > 0)[0]

    box_coords = box_coords[keep]
    box_cls = (box_cls[keep]).astype(np.int8)
    box_scores = box_scores[keep]

    # clip boxes inside image
    box_coords = clip_boxes(np.ascontiguousarray(box_coords, dtype=np.float32),
                            cfg.INP_SIZE, cfg.INP_SIZE)

    return box_coords, box_cls, box_scores
