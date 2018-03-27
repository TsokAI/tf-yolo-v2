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
        int out_w, int out_h):
    """
    Yolo2
    Transform predicted proposals to image bounding boxes, similar to bbox_transform_inv in faster-rcnn
    cython parallel with 4 threads
    ----------
    Parameters
    ----------
    bbox_pred: 3-dim float ndarray [out_h*out_w, num_anchors, 4] of (sig(tx), sig(ty), exp(th), exp(tw))
    anchors: [num_anchors, 2] of (ph, pw)
    out_w, out_h: width, height of feature map
    Returns
    -------
    box_pred: 3-dim float ndarray [out_h*out_w, num_anchors, 4] of bbox (x1, y1, x2, y2) rescaled to (0, 1)
    """
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] box_pred = np.zeros((out_h*out_w, num_anchors, 4), dtype=DTYPE)
    cdef DTYPE_t cx, cy, bw, bh
    cdef unsigned int row, col, a, ind
    
    for row in range(out_h):
        for col in range(out_w):
            ind = row * out_w + col
            for a in range(num_anchors):
                # box_pred in output's scale
                cx = bbox_pred[ind, a, 0] + col
                cy = bbox_pred[ind, a, 1] + row
                bw = anchors[a, 0] * bbox_pred[ind, a, 2] * 0.5
                bh = anchors[a, 1] * bbox_pred[ind, a, 3] * 0.5
                box_pred[ind, a, 0] = (cx - bw) / out_w
                box_pred[ind, a, 1] = (cy - bh) / out_h
                box_pred[ind, a, 2] = (cx + bw) / out_w
                box_pred[ind, a, 3] = (cy + bh) / out_h

    return box_pred

def bbox_transform_inv(
        np.ndarray[DTYPE_t, ndim=3] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int out_w, int out_h):
    
    return bbox_transform_inv_op(bbox_pred, anchors, out_w, out_h)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bbox_transform_inv3_op(
    np.ndarray[DTYPE_t, ndim=3] bbox_pred,
    np.ndarray[DTYPE_t, ndim=2] anchors,
    int in_w, int in_h, int out_w, int out_h):
    """
    Yolo3
    Transform predicted proposals to image bounding boxes, similar to bbox_transform_inv in faster-rcnn
    cython parallel with 4 threads
    ----------
    Parameters
    ----------
    bbox_pred: 3-dim float ndarray [out_h*out_w, num_anchors, 4] of (sig(tx), sig(ty), exp(th), exp(tw))
    anchors: [num_anchors, 2] of (ph, pw) in input's scale
    in_w, in_h: width, height of input
    out_w, out_h: width, height of feature map
    Returns
    -------
    box_pred: 3-dim float ndarray [out_h*out_w, num_anchors, 4] of bbox (x1, y1, x2, y2) rescaled to (0, 1)
    """
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] box_pred = np.zeros((out_h*out_w, num_anchors, 4), dtype=DTYPE)
    cdef DTYPE_t cx, cy, bw, bh
    cdef unsigned int row, col, a, ind
    cdef DTYPE_t xfeat = in_w / out_w
    cdef DTYPE_t yfeat = in_h / out_h

    for row in range(out_h):
        for col in range(out_w):
            ind = row * out_w + col
            for a in range(num_anchors):
                # cx, cy in output's scale, rescale to input's scale
                cx = (bbox_pred[ind, a, 0] + col) * xfeat
                cy = (bbox_pred[ind, a, 1] + row) * yfeat
                # bw, bh in input's scale
                bw = anchors[a, 0] * bbox_pred[ind, a, 2] * 0.5
                bh = anchors[a, 1] * bbox_pred[ind, a, 3] * 0.5
                box_pred[ind, a, 0] = cx - bw
                box_pred[ind, a, 1] = cy - bh
                box_pred[ind, a, 2] = cx + bw
                box_pred[ind, a, 3] = cy + bh

    return box_pred

def bbox_transform_inv3(
    np.ndarray[DTYPE_t, ndim=3] bbox_pred,
    np.ndarray[DTYPE_t, ndim=2] anchors,
    int in_w, int in_h, int out_w, int out_h):

    return bbox_transform_inv3_op(bbox_pred, anchors, in_w, in_h, out_w, out_h)

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
