from __future__ import absolute_import, division, print_function
import numpy as np

# default anchors in yolo2 416x416
default_anchors = np.asarray([(1.19, 1.08), (4.41, 3.42), (11.38, 6.63),
                              (5.11, 9.42), (10.52, 16.62)], dtype=np.float32)


def get_anchors(target_size):
    if target_size == (416, 416):
        return default_anchors

    target_scale = np.array(target_size, dtype=np.float32) / 416.
    return np.round(default_anchors * target_scale, 2)
