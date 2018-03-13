from __future__ import absolute_import, division, print_function
import numpy as np
from multiprocessing import Pool
from functools import partial
from layers.compute_targets import compute_targets_image

pool = Pool(processes=4)


def proposal_target_layer(bbox_pred, iou_pred, gt_boxes, gt_cls, anchors, logitsize, warmup=False):
    """Compute targets, masks for batch
    Parameters:
        bbox_pred: np.ndarray [batch_size, hw, num_anchors_cell, 4]
        iou_pred: np.ndarray [batch_size, hw, num_anchors_cell, 1]
        gt_boxes: np.ndarray [batch_size, num_gt_boxes, 4]
        gt_cls: np.ndarray [batch_size, num_gt_boxes]
        anchors: np.ndarray [num_anchors_cell, 2]
    Return:
        groundtruths, masks: np.ndarray [batch_size, hw, num_anchors_cell, codesize]
    """
    targets = pool.map(partial(compute_targets_image, anchors=anchors, logitsize=logitsize, warmup=warmup),
                       ((bbox_pred[i], iou_pred[i], gt_boxes[i], gt_cls[i])
                        for i in range(gt_boxes.shape[0])))

    bbox_target = np.stack(t[0] for t in targets)
    bbox_mask = np.stack(t[1] for t in targets)
    iou_target = np.stack(t[2] for t in targets)
    iou_mask = np.stack(t[3] for t in targets)
    cls_target = np.stack(t[4] for t in targets)
    cls_mask = np.stack(t[5] for t in targets)

    num_boxes = np.stack(t[6] for t in targets)
    num_boxes = np.sum(num_boxes).astype(np.float32)

    return bbox_target, bbox_mask, iou_target, iou_mask, cls_target, cls_mask, num_boxes
