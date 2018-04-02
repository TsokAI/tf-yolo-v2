# yolo2
# compute targets for regression/classification from proposal_target_layer
from __future__ import absolute_import, division, print_function
import numpy as np
from utils.bbox_transform import bbox_transform_inv
from utils.bbox import box_overlaps, anchor_overlaps
import config as cfg


def proposal_target_layer(feed_data, anchors, logitsize, warmup):
    bbox_pred, iou_pred, gt_boxes, gt_cls = feed_data

    # filter ignored groundtruth boxes
    gt_inds = np.where(gt_cls >= 0)[0]
    gt_boxes = gt_boxes[gt_inds]
    gt_cls = gt_cls[gt_inds]

    num_boxes = len(gt_inds)  # number of groundtruth boxes

    # transform bbox and rescale to inp_size
    box_pred = bbox_transform_inv(np.ascontiguousarray(bbox_pred, dtype=np.float32), np.ascontiguousarray(
        anchors, dtype=np.float32), logitsize, logitsize) * cfg.INP_SIZE

    hw, num_anchors, _ = box_pred.shape

    cls_target = np.zeros((hw, num_anchors, cfg.NUM_CLASSES), dtype=np.float32)
    cls_mask = np.zeros((hw, num_anchors), dtype=np.float32)
    iou_target = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    iou_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)
    bbox_target = np.zeros((hw, num_anchors, 4), dtype=np.float32)
    bbox_mask = np.zeros((hw, num_anchors, 1), dtype=np.float32)

    if warmup:  # match prediction boxes to anchors
        bbox_target[:, :, 0:2] = 0.5  # sig(tx), sig(ty) = 0.5 -> tx, ty = 0
        bbox_target[:, :, 2:4] = 1.0  # exp(tw), exp(th) = 1.0 -> tw, th = 0
        bbox_mask += cfg.COORD_SCALE  # regression all prediction boxes

    # compute overlaps btw prediction and groundtruth boxes
    box_pred = np.reshape(box_pred, [-1, 4])

    box_ious = box_overlaps(np.ascontiguousarray(box_pred, dtype=np.float32),
                            np.ascontiguousarray(gt_boxes, dtype=np.float32))  # nan

    box_ious = np.reshape(box_ious, [hw, num_anchors, -1])

    # select boxes with best iou smaller than thresh to assign negative
    neg_box_inds = np.where(np.max(box_ious, axis=2) < cfg.IOU_THRESH)
    iou_mask[neg_box_inds] = cfg.NO_OBJECT_SCALE

    # locate groundtruth cells, compute bbox target
    feat_stride = cfg.INP_SIZE / logitsize

    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5 / feat_stride
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5 / feat_stride
    cell_inds = np.floor(cy) * logitsize + np.floor(cx)
    cell_inds = cell_inds.astype(np.int32)

    # transform to bbox
    box_target = np.empty(gt_boxes.shape, dtype=np.float32)
    box_target[:, 0] = cx - np.floor(cx)
    box_target[:, 1] = cy - np.floor(cy)
    box_target[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / feat_stride
    box_target[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / feat_stride

    # select best anchor for each groundtruth boxes
    gt_boxes /= feat_stride  # rescale to logits' scale

    anchor_ious = anchor_overlaps(np.ascontiguousarray(
        anchors, dtype=np.float32), np.ascontiguousarray(gt_boxes, dtype=np.float32))

    anchor_inds = np.argmax(anchor_ious, axis=0)

    # compute targets, masks
    for i, cell_i in enumerate(cell_inds):
        if cell_i >= hw or cell_i < 0:  # skip gt outside logits
            continue

        a = anchor_inds[i]

        cls_target[cell_i, a, gt_cls[i]] = 1
        cls_mask[cell_i, a] = cfg.CLASS_SCALE[gt_cls[i]]  # weight of class

        iou_target[cell_i, a, :] = 1
        # iou_target[cell_i, a, :] = box_ious[cell_i, a, i]  # rescore iou target
        iou_mask[cell_i, a, :] = cfg.OBJECT_SCALE

        box_target[i, 2:4] /= anchors[a]
        bbox_target[cell_i, a, :] = box_target[i]
        bbox_mask[cell_i, a, :] = cfg.COORD_SCALE

    return num_boxes, cls_target, cls_mask, iou_target, iou_mask, bbox_target, bbox_mask
