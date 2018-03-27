from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
from config import LABEL_NAMES


def draw_targets(image, box_coords, box_cls, box_scores):
    for i in range(box_coords.shape[0]):
        p1 = (box_coords[i, 0], box_coords[i, 1])
        p2 = (box_coords[i, 2], box_coords[i, 3])

        cv2.rectangle(image, p1, p2, (0, 0, 255), 1)
        cv2.putText(image, LABEL_NAMES[box_cls[i]] + '_{:.3f}'.format(box_scores[i]),
                    p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return image
