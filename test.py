from __future__ import absolute_import, division, print_function
import os
from imdb import Imdb
from network import Network

data_dir = os.path.join(os.getcwd(), 'data')
anno_dir = os.path.join(data_dir, 'eval_annotation')
images_dir = os.path.join(data_dir, 'images')

imdb = Imdb(anno_dir, images_dir,
            batch_size=1)

net = Network(is_training=False)

for images, gt_boxes, gt_cls in imdb.next_batch():  # batch_size is 1
    box_pred, cls_inds, scores = net.predict(images)
