from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from functools import partial
import nets.resnet as net
import config3 as cfg
from layers.proposal_target_layer3 import proposal_target_layer
from layers.proposal_layer3 import proposal_layer
from nms.gpu_nms import gpu_nms
from utils.bbox_transform import clip_boxes

slim = tf.contrib.slim
pool = Pool(processes=4)

ckpt_dir = os.path.join(os.getcwd(), 'ckpt3')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


def build_targets(bbox_pred, iou_pred, gt_boxes, gt_cls, anchors, logitsize, warmup):
    targets = pool.map(partial(proposal_target_layer,
                               anchors=anchors, logitsize=logitsize, warmup=warmup),
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
        self.session = tf.Session()

        self.images_ph = tf.placeholder(
            tf.uint8, shape=[None, cfg.INP_SIZE, cfg.INP_SIZE, 3])

        # anchors pyramid, from yolo3 paper
        self.anchors = {
            'block4': tf.Variable([[116, 90], [156, 198], [373, 326]],
                                  trainable=False, name='anchor_block4', dtype=tf.float32),
            'block3': tf.Variable([[30, 61], [62, 45], [59, 119]],
                                  trainable=False, name='anchor_block3', dtype=tf.float32),
            'block2': tf.Variable([[10, 13], [16, 30], [33, 23]],
                                  trainable=False, name='anchor_block2', dtype=tf.float32)
        }

        # color distortion for traning
        preprocess_images = net.preprocess(self.images_ph)

        end_points = net.forward(
            preprocess_images, (cfg.NUM_ANCHORS_CELL*(5+cfg.NUM_CLASSES)), is_training)

        if is_training:
            self.gt_boxes_ph = tf.placeholder(tf.float32)
            self.gt_cls_ph = tf.placeholder(tf.int8)

            self.warmup = tf.placeholder(tf.bool)

            RSUM = tf.losses.Reduction.SUM

            self.bbox_loss = 0
            self.iou_loss = 0
            self.cls_loss = 0
        else:
            self.box_pred = []
            self.cls_inds = []
            self.scores = []

        for block in end_points:
            logits = end_points[block]
            logitsize = logits.get_shape()[1]  # NHWC tensor, H=W
            logits = tf.reshape(
                logits, shape=[-1, logitsize*logitsize, cfg.NUM_ANCHORS_CELL, 5 + cfg.NUM_CLASSES])

            # [sig(tx), sig(ty), exp(tw), exp(th)]
            xy_pred = tf.sigmoid(logits[:, :, :, 0:2])
            wh_pred = tf.exp(logits[:, :, :, 2:4])
            bbox_pred = tf.concat([xy_pred, wh_pred], axis=3)

            # sig(to)
            iou_pred = tf.sigmoid(logits[:, :, :, 4:5])

            cls_pred = logits[:, :, :, 5:]

            if is_training:
                bbox_target, bbox_mask, iou_target, iou_mask, cls_target, cls_mask, num_boxes = tf.py_func(
                    build_targets,
                    [bbox_pred, iou_pred,
                     self.gt_boxes_ph, self.gt_cls_ph, self.anchors[block], logitsize, self.warmup],
                    [tf.float32] * 7,
                    name=block+'_proposal_target_layer')

                self.bbox_loss += tf.losses.mean_squared_error(
                    bbox_target*bbox_mask, bbox_pred*bbox_mask, scope=block+'_bbox_loss', reduction=RSUM) / num_boxes

                self.iou_loss += tf.losses.mean_squared_error(
                    iou_target*iou_mask, iou_pred*iou_mask, scope=block+'_iou_loss', reduction=RSUM) / num_boxes

                cls_mask = tf.squeeze(cls_mask, axis=-1)
                self.cls_loss += tf.losses.softmax_cross_entropy(
                    cls_target, cls_pred, cls_mask, scope=block+'_cls_loss', reduction=RSUM) / num_boxes
            else:
                cls_pred = tf.nn.softmax(cls_pred)

                # batch_size must be 1
                box_pred, cls_inds, scores = tf.py_func(
                    proposal_layer,
                    [bbox_pred[0], iou_pred[0], cls_pred[0],
                        self.anchors[block], logitsize],
                    [tf.float32, tf.int8, tf.float32],
                    name=block+'_proposal_layer')

                self.box_pred.append(box_pred)
                self.cls_inds.append(cls_inds)
                self.scores.append(scores)

        if is_training:
            self.total_loss = self.bbox_loss + self.iou_loss + self.cls_loss

            self.global_step = tf.Variable(
                0, trainable=False, name='global_step')

            self.learning_rate = tf.train.exponential_decay(
                learning_rate, self.global_step, 12500, 0.75, staircase=True)

            self.optimizer = tf.train.MomentumOptimizer(
                self.learning_rate, 0.9).minimize(self.total_loss, self.global_step)

            # training summaries
            tf.summary.scalar('bbox_loss', self.bbox_loss)
            tf.summary.scalar('iou_loss', self.iou_loss)
            tf.summary.scalar('cls_loss', self.cls_loss)
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.merged = tf.summary.merge_all()

            self.writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), 'logs'), self.session.graph)
        else:
            self.box_pred = tf.concat(self.box_pred, axis=0)
            self.cls_inds = tf.concat(self.cls_inds, axis=0)
            self.scores = tf.concat(self.scores, axis=0)

        self.saver = tf.train.Saver(max_to_keep=1)

        # restore with ckpt/pretrain or init
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=ckpt_dir)
            self.saver.restore(self.session, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            print('init variables')
            # from slim pretrained model
            net.restore(self.session, tf.global_variables())

    def save_ckpt(self):
        self.saver.save(self.session,
                        save_path=os.path.join(ckpt_dir, cfg.DATASET),
                        global_step=self.global_step)

        print('new checkpoint saved')

    def fit(self, images, gt_boxes, gt_cls, warmup=False):  # training on batch
        summary, step, _ = self.session.run([self.merged, self.global_step,
                                             self.optimizer],
                                            feed_dict={self.images_ph: images,
                                                       self.gt_boxes_ph: gt_boxes,
                                                       self.gt_cls_ph: gt_cls,
                                                       self.warmup: warmup})

        self.writer.add_summary(summary, step)

    def predict(self, images):
        box_pred, cls_inds, scores = self.session.run([self.box_pred, self.cls_inds, self.scores],
                                                      feed_dict={self.images_ph: images})

        # apply nms and clip boxes
        keep = np.zeros(len(box_pred), dtype=np.int8)
        for i in range(cfg.NUM_CLASSES):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue

            dets = np.ascontiguousarray(
                np.hstack([box_pred[inds], scores[inds]]), dtype=np.float32)

            keep_in_cls = gpu_nms(dets, cfg.NMS_THRESH)

            keep[inds[keep_in_cls]] = 1

        keep = np.where(keep > 0)[0]

        box_pred = box_pred[keep]
        cls_inds = cls_inds[keep]
        scores = scores[keep][:, 0]

        box_pred = clip_boxes(np.ascontiguousarray(box_pred, dtype=np.float32),
                              cfg.INP_SIZE, cfg.INP_SIZE)

        cls_inds = cls_inds.astype(np.int8)

        return box_pred, cls_inds, scores


if __name__ == '__main__':
    Network(learning_rate=1e-3)