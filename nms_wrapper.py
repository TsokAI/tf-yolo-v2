from __future__ import absolute_import, division, print_function


def nms(dets, thresh, force_cpu=False):
    if dets.shape[0] == 0:
        return []
    if force_cpu:
        from nms.cpu_nms import cpu_nms
        return cpu_nms(dets, thresh)
    else:
        from nms.gpu_nms import gpu_nms
        return gpu_nms(dets, thresh)
