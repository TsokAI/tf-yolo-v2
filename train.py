from __future__ import absolute_import, division, print_function
import argparse
import os
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader
from utils.anchors import get_anchors

slim = tf.contrib.slim

train_anno_dir = os.path.join(cfg.data_dir, 'annotation_test')

# add gpu/cpu options??
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

# tf configuration
tfcfg = tf.ConfigProto()
tfcfg.gpu_options.per_process_gpu_memory_fraction = 0.8
tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=tfcfg), is_training=True,
              lr=args.lr, adamop=True, pretrained=True)

# load anchors and data
print('loading anchors and dataset')
anchors = get_anchors(target_size=(cfg.inp_size, cfg.inp_size))
blob = BlobLoader(anno_dir=train_anno_dir, batch_size=args.batch)
print('start training')

num_iters = blob.num_anno // args.batch
step = 0

for epoch in range(1, args.epochs + 1):
    iter = 0

    for batch_images, batch_boxes, batch_classes, num_boxes_batch in blob.next_batch():
        iter += 1

        step, loss = net.train(batch_images, batch_boxes,
                               batch_classes, anchors, num_boxes_batch)

        if step % 1 == 0 or iter == num_iters:
            print('step: {0:06} - total loss: {1}'.format(step, loss))

    if epoch % 10 == 0 or epoch == args.epochs:
        net.save_ckpt(step)
