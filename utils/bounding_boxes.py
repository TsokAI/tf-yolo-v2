from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
from config import LABEL_NAMES


def draw_targets(image, box_pred, cls_inds, scores):
    for i in range(box_pred.shape[0]):
        p1 = (box_pred[i, 1], box_pred[i, 0])
        p2 = (box_pred[i, 3], box_pred[i, 2])

        cv2.rectangle(image, p1, p2, (255, 255, 255), 1)
        cv2.putText(image, LABEL_NAMES[cls_inds[i]] + '_{:.3f}'.format(scores[i]),
                    p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return image
