# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef inline DTYPE_t max_c(DTYPE_t a, DTYPE_t b):
    return a if a >= b else b

cdef inline DTYPE_t min_c(DTYPE_t a, DTYPE_t b):
    return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef box_overlaps_op(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Compute overlaps of boxes and query_boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float in order (x1, y1, x2, y2)
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih
    cdef DTYPE_t box_area, inter_area, ua
    cdef unsigned int n, k
    for n in range(N):
        box_area = (
            (boxes[n, 2] - boxes[n, 0] + 1) *
            (boxes[n, 3] - boxes[n, 1] + 1)
        )
        for k in range(K):
            iw = (
                min_c(boxes[n, 2], query_boxes[k, 2]) -
                max_c(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min_c(boxes[n, 3], query_boxes[k, 3]) -
                    max_c(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    inter_area = iw * ih
                    ua = float(
                        (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                        (query_boxes[k, 3] - query_boxes[k, 1] + 1) +
                        box_area - inter_area
                    )
                    overlaps[n, k] = inter_area / ua
    return overlaps

def box_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    
    return box_overlaps_op(boxes, query_boxes)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef anchor_overlaps_op(
        np.ndarray[DTYPE_t, ndim=2] anchors,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    For each query box compute the intersection ratio covered by anchors
    ----------
    Parameters
    ----------
    anchors: (N, 2) ndarray of float in order (width, height)
    query_boxes: (K, 4) ndarray of float in order (x1, y1, x2, y2)
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    cdef unsigned int N = anchors.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, boxw, boxh
    cdef DTYPE_t anchor_area, inter_area
    cdef unsigned int n, k
    for n in range(N):
        anchor_area = anchors[n, 0] * anchors[n, 1]
        for k in range(K):
            boxw = query_boxes[k, 2] - query_boxes[k, 0] + 1
            boxh = query_boxes[k, 3] - query_boxes[k, 1] + 1
            iw = min_c(anchors[n, 0], boxw)
            ih = min_c(anchors[n, 1], boxh)
            inter_area = iw * ih
            overlaps[n, k] = inter_area / (anchor_area + boxw * boxh - inter_area)
    return overlaps

def anchor_overlaps(
        np.ndarray[DTYPE_t, ndim=2] anchors,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):

    return anchor_overlaps_op(anchors, query_boxes)
