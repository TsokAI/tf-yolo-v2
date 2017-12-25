# tf-yolo2 configurations
from __future__ import absolute_import, division, print_function
import os
import numpy as np

# feature extractor
# vgg_16, MobilenetV1, 
model = 'vgg_16'

# working directories, create symlink to 'data' folder
# contain 'annotation' and 'images' subfolder
workspace = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(workspace, 'data')

# todo: various shapes (inp_h != inp_w)
# inp_size = [384, 416, 448, 480, 512]
inp_size = (416, 416)

# object labels and class colors
# pascal labels
# label_names = ['person',
#                'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#                'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#                'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']

# detrac labels
label_names = ['others', 'car', 'bus', 'van']

num_classes = len(label_names)
label_colors = {}
for label in label_names:
    label_colors[label] = (np.random.randint(
        50, 128), np.random.randint(50, 128), np.random.randint(50, 128))

# configuration
iou_thresh = 0.7
cls_scale = 1
object_scale = 5
noobject_scale = 1
box_scale = 1

# anchors with (height, width) order
anchors = np.array([(1.19, 1.08), (4.41, 3.42), (11.38, 6.63),
                    (5.11, 9.42), (10.52, 16.62)], dtype=np.float32)

num_anchors = len(anchors)
