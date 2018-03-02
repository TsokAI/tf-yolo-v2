from __future__ import absolute_import, division, print_function
import numpy as np

# anchors [height, width] with inp_size is 416
default_anchors = np.array([[1.08, 1.19],
                            [3.42, 4.41],
                            [6.63, 11.38],
                            [9.42, 5.11],
                            [16.62, 10.52]], dtype=np.float32)


def generate_anchors(inp_size=416):
    if inp_size == 416:
        return default_anchors

    anchors = np.round(default_anchors * inp_size / 416., 2)

    return anchors
