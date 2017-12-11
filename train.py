from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader
from utils.anchors import get_anchors

slim = tf.contrib.slim

train_anno_dir = os.path.join(cfg.data_dir, 'annotation')

# add gpu/cpu options??
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
blob = BlobLoader(anno_dir=train_anno_dir, batch_size=args.batch)
print('done')

# gpu, jit/xla config
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.9
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# also load checkpoint or init variables
net = Network(session=tf.Session(config=tfcfg),
              lr=args.lr, adamop=True, pretrained=True)

for epoch in range(1, args.epochs + 1):
    for _ in range(blob.num_anno // args.batch + 1):
        batch_images, batch_boxes, batch_classes = blob.next_batch()
        step, loss = net.train(batch_images, batch_boxes,
                               batch_classes, anchors)

        if step % 5000 == 0:
            print('step: {0:06} - total loss: {1:.6f}'.format(step, loss))

    if epoch % 10 == 0 or epoch == args.epochs:
        net.save_ckpt()
