from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2
from network import Network
from utils.bounding_boxes import draw_targets
from config import INP_SIZE
import time
from datetime import timedelta


images_dir = os.path.join(os.getcwd(), 'test')

output_dir = os.path.join(images_dir, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

net = Network(is_training=False)

test_t = time.time()

for i in os.listdir(images_dir):
    if i == 'output':
        continue

    image = cv2.imread(os.path.join(images_dir, i))

    image = cv2.resize(image, (INP_SIZE, INP_SIZE))

    image_cp = np.copy(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # batch_size is 1
    box_pred, cls_inds, scores = net.predict(image[np.newaxis])

    image_cp = draw_targets(image_cp, box_pred, cls_inds, scores)

    cv2.imwrite(os.path.join(output_dir, i), image_cp)

print('testing done - time: {}'.format(str(timedelta(seconds=time.time() - test_t))))
