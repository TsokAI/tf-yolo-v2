# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md
from __future__ import absolute_import, division, print_function
import numpy as np
from config import num_classes
from utils.bbox import box_overlaps, box_intersections


def evaluate_image(box_pred, cls_pred, gt_boxes, gt_classes, thresh=0.5):
    """Compute avg_iou, precision, recall on image
    """
    avg_iou = 0
    precision = 0
    recall = 0

    for c in range(num_classes):
        gt_inds_c = np.where(gt_classes == c)[0]
        pred_inds_c = np.where(cls_pred == c)[0]

        if len(gt_inds_c) == 0 or len(pred_inds_c) == 0:
            continue

        gt_boxes_c = gt_boxes[gt_inds_c]
        box_pred_c = box_pred[pred_inds_c]

        # 3 labels: true_positives, ignored, false_positives
        # true_positives, largest IoU per gt_boxes is greater than thresh (default=0.5)
        overlaps = np.max(box_overlaps(box_pred_c, gt_boxes_c), axis=0)

        true_pos = sum(overlaps > thresh)
        avg_iou += np.mean(overlaps)

        # ignored (group-of groundtruth box), area of intersection divided by
        # area of prediction is greater than 0.5
        # false_positives, neither true_positives nor ignored
        intersections = box_intersections(box_pred_c, gt_boxes_c) > 0.5

        false_neg = len(np.where(np.sum(intersections, axis=0) == 0)[0])
        false_pos = len(np.where(np.sum(intersections, axis=1) == 0)[0])

        # precision = num-of-true_positives / (num-of-true_positives + num-of-false_positives)
        # recall = num-of-true_positives / num-of-non-group-of-boxes
        # non-group-of-boxes (#gt_boxes, box_intersections <= 0.5)
        precision += true_pos / (true_pos + false_pos)
        recall += true_pos / (true_pos + false_neg)

    return avg_iou / num_classes, precision / num_classes, recall / num_classes
