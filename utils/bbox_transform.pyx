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
cdef bbox_transform_inv_op(
        np.ndarray[DTYPE_t, ndim=3] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int H, int W):
    """
    Transform predicted proposals to image bounding boxes, similar to bbox_transform_inv in faster-rcnn
    cython parallel with 4 threads
    ----------
    Parameters
    ----------
    bbox_pred: 3-dim float ndarray [HxW, num_anchors, 4] of (sig(tx), sig(ty), exp(th), exp(tw))
    anchors: [num_anchors, 2] of (ph, pw)
    H, W: height, width of feature map
    Returns
    -------
    box_pred: 3-dim float ndarray [HxW, num_anchors, 4] of bbox (x1, y1, x2, y2) rescaled to (0, 1)
    """
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] box_pred = np.zeros((H*W, num_anchors, 4), dtype=DTYPE)
    cdef DTYPE_t cx, cy, bh, bw
    cdef unsigned int row, col, a, ind
    
    for row in range(H):
        for col in range(W):
            ind = row * W + col
            for a in range(num_anchors):
                cx = bbox_pred[ind, a, 0] + col
                cy = bbox_pred[ind, a, 1] + row
                bw = anchors[a, 0] * bbox_pred[ind, a, 2] * 0.5
                bh = anchors[a, 1] * bbox_pred[ind, a, 3] * 0.5
                box_pred[ind, a, 0] = (cx - bw) / W
                box_pred[ind, a, 1] = (cy - bh) / H
                box_pred[ind, a, 2] = (cx + bw) / W
                box_pred[ind, a, 3] = (cy + bh) / H

    return box_pred

def bbox_transform_inv(
        np.ndarray[DTYPE_t, ndim=3] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int H, int W):
    
    return bbox_transform_inv_op(bbox_pred, anchors, H, W)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef clip_boxes_op(
    np.ndarray[DTYPE_t, ndim=2] boxes,
    int H, int W):

    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int n

    for n in range(N):
        boxes[n, 0] = max_c(min_c(boxes[n, 0], W), 0)
        boxes[n, 1] = max_c(min_c(boxes[n, 1], H), 0)
        boxes[n, 2] = max_c(min_c(boxes[n, 2], W), 0)
        boxes[n, 3] = max_c(min_c(boxes[n, 3], H), 0)

    return boxes

def clip_boxes(
    np.ndarray[DTYPE_t, ndim=2] boxes,
    int H, int W):

    return clip_boxes_op(boxes, H, W)