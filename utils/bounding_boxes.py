from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
from config import LABEL_NAMES


def draw_targets(image, box_pred, cls_inds, scores):
    for b in range(box_pred.shape[0]):
        p1 = (box_pred[b, 1], box_pred[b, 0])
        p2 = (box_pred[b, 3], box_pred[b, 2])

        cv2.rectangle(image, p1, p2, (255, 255, 255), 1)
        cv2.putText(image, '{}_{:.3f}'.format(
            LABEL_NAMES[cls_inds[b]], scores[b]), p1,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return image
