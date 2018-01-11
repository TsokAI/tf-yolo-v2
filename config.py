# tf-yolo2 configurations
from __future__ import absolute_import, division, print_function
import os
import numpy as np

# working directories, create symlink to 'data' folder
# contain 'annotation' and 'images' subfolder

# pascal/voc labels
label_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

# detrac labels
# label_names = ['others', 'car', 'bus', 'van']

num_classes = len(label_names)

# model = 'vgg'
model = 'resnet'
# model = 'mobilenet'

# inp_size = [384, 416, 448, 480, 512]
inp_size = (416, 416)

# yolov2 configuration
iou_thresh = 0.6
cls_scale = 1
object_scale = 5
noobject_scale = 1
box_scale = 1

nms_thresh = 0.45

# anchors with (height, width) order
default_anchors = np.array([(1.19, 1.08),
                            (4.41, 3.42),
                            (11.38, 6.63),
                            (5.11, 9.42),
                            (10.52, 16.62)], dtype=np.float32)

num_anchors = len(anchors)
