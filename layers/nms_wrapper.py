from __future__ import absolute_import, division, print_function
from nms.cpu_nms import cpu_nms
from nms.gpu_nms import gpu_nms


def nms_detection(dets, thresh, use_gpu=True):
    if use_gpu:
        return gpu_nms(dets, thresh)
    else:
        return cpu_nms(dets, thresh)
