from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
import re

with open('./voc07vgg_16.txt', 'r') as f:
    data = f.readlines()

data = [x.strip() for x in data if re.match('step', x)]

step = []
bbox_loss = []
iou_loss = []
cls_loss = []

for x in data:
    x = x.split('-')

    step.append(int(x[0].split(':')[1]))
    bbox_loss.append(float(x[1].split(':')[1]))
    iou_loss.append(float(x[2].split(':')[1]))
    cls_loss.append(float(x[3].split(':')[1]))

step = np.array(step, dtype=np.int)
bbox_loss = np.array(bbox_loss, dtype=np.float32)
iou_loss = np.array(iou_loss, dtype=np.float32)
cls_loss = np.array(cls_loss, dtype=np.float32)
