# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md
from __future__ import absolute_import, division, print_function
import numpy as np
import config as cfg
from utils.bbox import box_overlaps, box_intersections


def evaluate(pred_boxes, pred_classes, gt_boxes, gt_classes, thresh=0.5):
    """Compute avg_iou, precision, recall
    """
    mean_iou = 0
    precision = 0
    recall = 0
    for c in range(cfg.num_classes):
        gt_inds_c = np.where(gt_classes == c)[0]
        pred_inds_c = np.where(pred_classes == c)[0]
        gt_boxes_c = gt_boxes[gt_inds_c]
        pred_boxes_c = pred_boxes[pred_inds_c]

        # 3 labels: true_positives, ignored, false_positives
        # true_positives, largest IoU per gt_boxes is greater than thresh (default=0.5)
        overlaps = box_overlaps(pred_boxes_c, gt_boxes_c)

        true_pos = sum(np.max(overlaps, axis=0) > thresh)
        mean_iou += np.mean(overlaps)

        # ignored (group-of groundtruth box), area of intersection divided by
        # area of prediction is greater than 0.5
        # false_positives, neither true_positives nor ignored
        overlaps = box_intersections(pred_boxes_c, gt_boxes_c) > 0.5

        false_neg = len(np.where(np.sum(overlaps, axis=0) == 0)[0])
        false_pos = len(np.where(np.sum(overlaps, axis=1) == 0)[0])

        # precision = num-of-true_positives / (num-of-true_positives + num-of-false_positives)
        # recall = num-of-true_positives / num-of-non-group-of-boxes
        # non-group-of-boxes (#gt_boxes, box_intersections <= 0.5)
        precision += true_pos / (true_pos + false_pos)
        recall += true_pos / (true_pos + false_neg)

    return mean_iou / cfg.num_classes, precision / cfg.num_classes, recall / cfg.num_classes
