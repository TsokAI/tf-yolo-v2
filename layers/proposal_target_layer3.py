# yolo3
# compute targets for regression/classification from proposal_target_layer
from __future__ import absolute_import, division, print_function
import numpy as np
from utils.bbox_transform import bbox_transform_inv3
from utils.bbox import box_overlaps, anchor_overlaps
import config as cfg


def proposal_target_layer(feed_data, anchors, out_w, out_h, warmup):
    bbox_pred, iou_pred, gt_boxes, gt_cls = feed_data

    # filter ignored groundtruth boxes
    gt_inds = np.where(gt_cls >= 0)[0]
    gt_boxes = gt_boxes[gt_inds]
    gt_cls = gt_cls[gt_inds]

    num_boxes = len(gt_inds)  # number of groundtruth boxes

    # transform bbox to input's scale boxes
    box_pred = bbox_transform_inv3(
        np.ascontiguousarray(bbox_pred, dtype=np.float32),
        np.ascontiguousarray(anchors, dtype=np.float32),
        cfg.INP_SIZE, cfg.INP_SIZE,  # square-image as input
        out_w, out_h)

    box_pred = np.reshape(box_pred, [-1, 4])

    hw = out_w * out_h
    cls_target = np.zeros(
        (hw, cfg.NUM_ANCHORS_CELL, cfg.NUM_CLASSES), dtype=np.float32)
    cls_mask = np.zeros((hw, cfg.NUM_ANCHORS_CELL), dtype=np.float32)
    iou_target = np.zeros((hw, cfg.NUM_ANCHORS_CELL, 1), dtype=np.float32)
    iou_mask = np.zeros((hw, cfg.NUM_ANCHORS_CELL, 1), dtype=np.float32)
    bbox_target = np.zeros((hw, cfg.NUM_ANCHORS_CELL, 4), dtype=np.float32)
    bbox_mask = np.zeros((hw, cfg.NUM_ANCHORS_CELL, 1), dtype=np.float32)

    if warmup:  # centering bbox, similar to yolo2
        bbox_mask += cfg.COORD_SCALE
        bbox_target[:, :, 0:2] = 0.5
        bbox_target[:, :, 2:4] = 1.0

    box_ious = box_overlaps(np.ascontiguousarray(box_pred, dtype=np.float32),
                            np.ascontiguousarray(gt_boxes, dtype=np.float32))  # nan

    box_ious = np.reshape(box_ious, [hw, cfg.NUM_ANCHORS_CELL, -1])

    negative_bbox_inds = np.where(np.max(box_ious, axis=2) < cfg.IOU_THRESH)
    iou_mask[negative_bbox_inds] = cfg.NO_OBJECT_SCALE * \
        (0 - iou_pred[negative_bbox_inds])

    # locating gt_boxes' cells in output's scale
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5 * (out_w / cfg.INP_SIZE)
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5 * (out_h / cfg.INP_SIZE)
    cell_inds = (np.floor(cy) * out_w + np.floor(cx)).astype(np.int32)

    # transform to bbox
    # nan at bbox if using resnet
    box_target = np.empty_like(gt_boxes, dtype=np.float32)
    box_target[:, 0] = cx - np.floor(cx)  # output's scale
    box_target[:, 1] = cy - np.floor(cy)
    box_target[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]  # input's scale
    box_target[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]

    # select best anchor as positive bounding box
    anchor_ious = anchor_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                  np.ascontiguousarray(gt_boxes, dtype=np.float32))

    best_anchor_inds = np.argmax(anchor_ious, axis=0)

    # compute targets
    for i, ci in enumerate(cell_inds):
        if ci < 0 or ci >= hw:
            continue  # skip cells outside output's size

        a = best_anchor_inds[i]

        cls_target[ci, a, gt_cls[i]] = 1
        cls_mask[ci, a] = cfg.CLASS_SCALE[gt_cls[i]]  # weight of class

        iou_target[ci, a, :] = 1
        iou_mask[ci, a, :] = cfg.OBJECT_SCALE * (1 - iou_pred[ci, a, :])

        box_target[i, 2:4] /= anchors[a]
        bbox_target[ci, a, :] = box_target[i]
        bbox_mask[ci, a, :] = cfg.COORD_SCALE

    return num_boxes, cls_target, cls_mask, iou_target, iou_mask, bbox_target, bbox_mask
