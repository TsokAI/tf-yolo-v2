from __future__ import absolute_import, division, print_function
import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import config as cfg
from blob import BlobLoader
from network import Network
from py_postprocess import postprocess
from py_evaluate import evaluate_image

slim = tf.contrib.slim

xla = tf.ConfigProto()
xla.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=xla),
              im_shape=cfg.inp_size, is_training=False)

data_dir = os.path.join(os.getcwd(), 'data')
anno_dir = os.path.join(data_dir, 'validation')
images_dir = os.path.join(data_dir, 'images')

blob = BlobLoader(anno_dir=anno_dir,
                  images_dir=images_dir,
                  batch_size=6, target_size=cfg.inp_size)

anchors = np.round(cfg.default_anchors * cfg.inp_size / 416, 2)

start_t = 0

avg_iou = 0
precision = 0
recall = 0

for batch_images, batch_boxes, batch_classes, _ in blob.next_batch():
    box_pred, iou_pred, cls_pred = net.predict(scaled_images=batch_images,
                                               anchors=anchors)

    for b in range(box_pred.shape[0]):
        box_pred_b, cls_inds_b, _ = postprocess(box_pred[b], iou_pred[b], cls_pred[b],
                                                im_shape=cfg.inp_size, thresh=0.5,
                                                force_cpu=False)

        gt_box_inds_b = np.where(batch_classes[b] >= 0)[0]
        gt_boxes_b = batch_boxes[b][gt_box_inds_b]
        gt_classes_b = batch_classes[b][gt_box_inds_b]

        avg_iou_b, precision_b, recall_b = evaluate_image(box_pred=box_pred_b,
                                                          cls_pred=cls_inds_b,
                                                          gt_boxes=gt_boxes_b,
                                                          gt_classes=gt_classes_b,
                                                          thresh=0.5)

        avg_iou += avg_iou_b
        precision += precision_b
        recall += recall_b

print('testing done - time: ' +
      str(timedelta(seconds=np.round(time.time() - start_t))))

num_images = blob.num_anno

print('num_images: {} - avg_iou: {} - precision: {} - recall: {}'.format(num_images,
                                                                         avg_iou / num_images,
                                                                         precision / num_images,
                                                                         recall / num_images))
