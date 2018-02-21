cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bbox_transform_op(
        np.ndarray[DTYPE_t, ndim=4] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int H, int W):
    """
    Transform predicted proposals to image bounding boxes, similar to bbox_transform_inv in faster-rcnn
    cython parallel with 4 threads
    ----------
    Parameters
    ----------
    bbox_pred: 4-dim float ndarray [bsize, HxW, num_anchors, 4] of (sig(tx), sig(ty), exp(th), exp(tw))
    anchors: [num_anchors, 2] of (ph, pw)
    H, W: height, width of feature map
    Returns
    -------
    box_pred: 4-dim float ndarray [bsize, HxW, num_anchors, 4] of bbox (x1, y1, x2, y2) rescaled to (0, 1)
    """
    cdef unsigned int bsize = bbox_pred.shape[0]
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=4] box_pred = np.zeros((bsize, H*W, num_anchors, 4), dtype=DTYPE)
    cdef DTYPE_t cx, cy, bh, bw
    cdef unsigned int b, row, col, a, ind
    
    for b in range(bsize):
        for row in range(H):
            for col in range(W):
                ind = row * W + col
                for a in range(num_anchors):
                    cx = bbox_pred[b, ind, a, 0] + row
                    cy = bbox_pred[b, ind, a, 1] + col
                    bh = anchors[a, 0] * bbox_pred[b, ind, a, 2] * 0.5
                    bw = anchors[a, 1] * bbox_pred[b, ind, a, 3] * 0.5
                    box_pred[b, ind, a, 0] = (cx - bh) / H
                    box_pred[b, ind, a, 1] = (cy - bw) / W
                    box_pred[b, ind, a, 2] = (cx + bh) / H
                    box_pred[b, ind, a, 3] = (cy + bw) / W

    return box_pred

@cython.boundscheck(False)
@cython.wraparound(False)
cdef clip_boxes_op(
    np.ndarray[DTYPE_t, ndim=2] boxes,
    int H, int W):

    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int n

    for n in range(N):
        boxes[n, 0] = max(min(boxes[n, 0], H - 1), 0)
        boxes[n, 1] = max(min(boxes[n, 1], W - 1), 0)
        boxes[n, 2] = max(min(boxes[n, 2], H - 1), 0)
        boxes[n, 3] = max(min(boxes[n, 3], W - 1), 0)

    return boxes

def bbox_transform(
        np.ndarray[DTYPE_t, ndim=4] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, 
        int H, int W):
    
    return bbox_transform_op(bbox_pred, anchors, H, W)

def clip_boxes(
    np.ndarray[DTYPE_t, ndim=2] boxes,
    int H, int W):

    return clip_boxes_op(boxes, H, W)
