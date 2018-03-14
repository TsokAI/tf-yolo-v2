from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
import config as cfg
import nets.resnet as net
from multiprocessing import Pool
from functools import partial
from layers.generate_anchors import generate_anchors
from layers.proposal_target_layer import proposal_target_layer
from layers.proposal_layer import proposal_layer

slim = tf.contrib.slim
pool = Pool(processes=4)

ckpt_dir = os.path.join(os.getcwd(), 'ckpt')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


def build_targets(bbox_pred, iou_pred, gt_boxes, gt_cls, anchors, logitsize, warmup):
    targets = pool.map(partial(proposal_target_layer, anchors=anchors, logitsize=logitsize, warmup=warmup),
                       ((bbox_pred[i], iou_pred[i], gt_boxes[i], gt_cls[i])
                        for i in range(gt_boxes.shape[0])))

    bbox_target = np.stack(t[0] for t in targets)
    bbox_mask = np.stack(t[1] for t in targets)
    iou_target = np.stack(t[2] for t in targets)
    iou_mask = np.stack(t[3] for t in targets)
    cls_target = np.stack(t[4] for t in targets)
    cls_mask = np.stack(t[5] for t in targets)
    num_boxes = np.sum(np.stack(t[6] for t in targets)).astype(np.float32)

    return bbox_target, bbox_mask, iou_target, iou_mask, cls_target, cls_mask, num_boxes


class Network(object):
    def __init__(self, is_training=True, learning_rate=None):
        self.sess = tf.Session()

        # [batch_size, inp_size, inp_size, channels]
        self.images_ph = tf.placeholder(
            tf.uint8, shape=[None, cfg.INP_SIZE, cfg.INP_SIZE, 3])

        self.warmup = tf.placeholder(tf.bool)

        # generate anchors for inp_size
        self.anchors = tf.Variable(generate_anchors(
            cfg.INP_SIZE), trainable=False, name='anchors')

        # TODO: freeze layers from pretrained model
        logits = net.forward(net.preprocess(self.images_ph), is_training)

        logits = slim.conv2d(logits, cfg.NUM_ANCHORS_CELL*(5+cfg.NUM_CLASSES),
                             [1, 1], activation_fn=None, scope='logits')

        logitsize = logits.get_shape().as_list()[1]  # NHWC tensor

        logits = tf.reshape(
            logits, shape=[-1, logitsize*logitsize, cfg.NUM_ANCHORS_CELL, 5 + cfg.NUM_CLASSES])

        # [sig(tx), sig(ty), exp(th), exp(tw)] for bbox prediction
        xy_pred = tf.sigmoid(logits[:, :, :, 0:2])
        wh_pred = tf.exp(logits[:, :, :, 2:4])
        bbox_pred = tf.concat([xy_pred, wh_pred], axis=3)

        # sig(to) for iou (predition-groundtruth) prediction
        iou_pred = tf.sigmoid(logits[:, :, :, 4:5])

        cls_pred = logits[:, :, :, 5:]

        if is_training:
            # training placeholders
            self.gt_boxes_ph = tf.placeholder(tf.float32)

            self.gt_cls_ph = tf.placeholder(tf.int8)

            # compute targets regression
            bbox_target, bbox_mask, iou_target, iou_mask, cls_target, cls_mask, num_boxes = tf.py_func(
                build_targets,
                [bbox_pred, iou_pred,
                 self.gt_boxes_ph, self.gt_cls_ph, self.anchors, logitsize, self.warmup],
                [tf.float32] * 7,
                name='proposal_target_layer')

            RSUM = tf.losses.Reduction.SUM

            # TODO: replace with smooth_l1 loss [model/objects-detection]
            self.bbox_loss = tf.losses.mean_squared_error(
                bbox_target*bbox_mask, bbox_pred*bbox_mask, scope='bbox_loss', reduction=RSUM) / num_boxes
            tf.summary.scalar('bbox_loss', self.bbox_loss)

            self.iou_loss = tf.losses.mean_squared_error(
                iou_target*iou_mask, iou_pred*iou_mask, scope='iou_loss', reduction=RSUM) / num_boxes
            tf.summary.scalar('iou_loss', self.iou_loss)

            # TODO: replace with softmax_focal_cross_entropy [model/objects-detection] [fbrs/detectron]
            self.cls_loss = tf.losses.softmax_cross_entropy(
                cls_target*cls_mask, cls_pred*cls_mask, scope='cls_loss', reduction=RSUM) / num_boxes
            tf.summary.scalar('cls_loss', self.cls_loss)

            self.total_loss = self.bbox_loss + self.iou_loss + self.cls_loss
            tf.summary.scalar('total_loss', self.total_loss)

            # training
            self.global_step = tf.Variable(
                0, trainable=False, name='global_step')

            self.learning_rate = tf.train.exponential_decay(
                learning_rate, self.global_step, 1500, 0.9, staircase=True)  # 6500

            self.optimizer = tf.train.MomentumOptimizer(
                self.learning_rate, 0.9).minimize(self.total_loss, self.global_step)

        else:
            # testing, batch_size is 1
            cls_pred = tf.nn.softmax(cls_pred)

            self.box_pred, self.cls_inds, self.scores = tf.py_func(
                proposal_layer,
                [bbox_pred[0], iou_pred[0], cls_pred[0], self.anchors, logitsize],
                [tf.float32, tf.int8, tf.float32],
                name='proposal_layer')

        self.saver = tf.train.Saver(max_to_keep=1)

        # summaries
        self.merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), 'logs'), self.sess.graph)

        # restore with ckpt/pretrain or init
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=ckpt_dir)
            self.saver.restore(self.sess, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            print('init variables')
            # from slim pretrained model
            net.restore(self.sess, tf.global_variables())

    def save_ckpt(self):
        self.saver.save(self.sess,
                        save_path=os.path.join(ckpt_dir, cfg.DATASET),
                        global_step=self.global_step)

        print('new checkpoint saved')

    def fit(self, images, gt_boxes, gt_cls, warmup=False):  # training on batch
        summary, step, _ = self.sess.run([self.merged, self.global_step,
                                          self.optimizer],
                                         feed_dict={self.images_ph: images,
                                                    self.gt_boxes_ph: gt_boxes,
                                                    self.gt_cls_ph: gt_cls,
                                                    self.warmup: warmup})

        self.writer.add_summary(summary, step)

    def predict(self, images):
        # batch_size must be 1
        box_pred, cls_inds, scores = self.sess.run([self.box_pred, self.cls_inds, self.scores],
                                                   feed_dict={self.images_ph: images})

        return box_pred, cls_inds, scores
