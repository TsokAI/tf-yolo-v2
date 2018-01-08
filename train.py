from __future__ import absolute_import, division, print_function
import argparse
import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader

slim = tf.contrib.slim

data_dir = os.path.join(os.getcwd(), 'data')
train_anno_dir = os.path.join(data_dir, 'annotation')
train_images_dir = os.path.join(data_dir, 'images')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learn_rate', type=float, default=1e-3)
args = parser.parse_args()

print('num_epochs: {0} - batch_size: {1} - learn_rate: {2}'.format(
    args.num_epochs, args.batch_size, args.learn_rate))

# tf configuration
xla = tf.ConfigProto()
xla.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=xla), im_shape=cfg.inp_size, is_training=True,
              lr=args.learn_rate, adamop=True, pretrained=True)

print('loading dataset')
blob = BlobLoader(anno_dir=train_anno_dir,
                  images_dir=train_images_dir,
                  batch_size=args.batch_size, target_size=cfg.inp_size)

anchors = np.round(cfg.anchors * cfg.inp_size / 416, 2)

# losses collection contain dict of losses from network
# {'step', 'bbox_loss', 'iou_loss', 'cls_loss', 'total_loss'}
losses_collection = []

step = 0
train_t = 0

print('start training')
for epoch in range(1, args.num_epochs + 1):
    start_t = time.time()

    for batch_images, batch_boxes, batch_classes, num_boxes_batch in blob.next_batch():

        step, bbox_loss, iou_loss, cls_loss = net.train(batch_images, batch_boxes, batch_classes,
                                                        anchors, num_boxes_batch)

        if step % 100 == 0:
            total_loss = bbox_loss + iou_loss + cls_loss

            losses_collection.append(
                (step, bbox_loss, iou_loss, cls_loss, total_loss))

            print('epoch: {0:03} - step: {1:07} - bbox_loss: {2} - iou_loss: {3} - cls_loss: {4}'
                  .format(epoch, step, bbox_loss, iou_loss, cls_loss))

    time_dif = np.round(time.time() - start_t)
    train_t += time_dif

    print('epoch: {0:03} - time: '.format(epoch) +
          str(timedelta(seconds=time_dif)))

    if epoch % 5 == 0 or epoch == args.num_epochs:
        net.save_ckpt(step)

print('training done - time: ' + str(timedelta(seconds=train_t)))

# dumpt collections to files
losses_collection = np.asarray(losses_collection, dtype=np.float32)
np.savetxt('./logs/losses_collection.txt', losses_collection, fmt='%.6e')
