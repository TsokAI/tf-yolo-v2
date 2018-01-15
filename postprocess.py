from __future__ import absolute_import, division, print_function
import numpy as np
from config import num_classes, nms_thresh
from nms_wrapper import nms
from utils.bbox import bbox_transform


def clip_boxes(boxes, im_shape):
    # Clip boxes[xmin, ymin, xmax, ymax] to image boundaries
    if boxes.shape[0] == 0:
        return boxes
    # 0 <= x1 < im_shape[0]
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[0] - 1), 0)
    # 0 <= y1 < im_shape[1]
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[1] - 1), 0)
    # 0 <= x2 < im_shape[0]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[0] - 1), 0)
    # 0 <= y2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[1] - 1), 0)

    return boxes


def postprocess(box_pred, iou_pred, cls_pred,
                im_shape, thresh, force_cpu=False):
    # flatten logits' cells
    box_pred = np.reshape(box_pred, newshape=[-1, 4])
    box_pred[:, 0::2] *= float(im_shape[0])
    box_pred[:, 1::2] *= float(im_shape[1])
    box_pred = box_pred.astype(np.int)

    iou_pred = np.reshape(iou_pred, newshape=[-1])
    cls_pred = np.reshape(cls_pred, newshape=[-1, num_classes])

    cls_inds = np.argmax(cls_pred, axis=1)
    scores = iou_pred * cls_pred[np.arange(cls_pred.shape[0]), cls_inds]

    # select positive boxes with score larger than thresh
    keep_inds = np.where(scores > thresh)[0]
    box_pred = box_pred[keep_inds]
    cls_inds = cls_inds[keep_inds]
    scores = scores[keep_inds]

    # apply nms to remove overlap boxes
    keep_inds = np.zeros(len(box_pred), dtype=np.int)
    for c in range(num_classes):
        inds = np.where(cls_inds == c)[0]
        if len(inds) == 0:
            continue

        dets = np.hstack((box_pred[inds], scores[inds][:, np.newaxis]))

        keep = nms(np.ascontiguousarray(dets, dtype=np.float32),
                   nms_thresh, force_cpu)
        keep_inds[inds[keep]] = 1

    keep_inds = np.where(keep_inds > 0)[0]
    box_pred = box_pred[keep_inds]
    cls_inds = cls_inds[keep_inds]
    scores = scores[keep_inds]

    box_pred = clip_boxes(box_pred, im_shape)

    return box_pred, cls_inds, scores
