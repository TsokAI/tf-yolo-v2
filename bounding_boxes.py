from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
from config import label_names, num_classes

label_colors = [(np.random.randint(0, 256),
                 np.random.randint(0, 256),
                 np.random.randint(0, 256)) for c in range(num_classes)]


def draw_targets(image, box_pred, cls_inds, scores):
    for b in range(box_pred.shape[0]):
        box_cls = cls_inds[b]

        box_label = label_names[box_cls]
        box_color = label_colors[box_cls]
        p1 = (box_pred[b, 1], box_pred[b, 0])
        p2 = (box_pred[b, 3], box_pred[b, 2])

        cv2.rectangle(image, p1, p2, box_color, 1)
        cv2.putText(image, '{}_{:.3f}'.format(
            box_label, scores[b]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color)

    return image
