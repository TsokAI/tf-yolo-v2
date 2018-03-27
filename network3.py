from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from functools import partial
import nets.resnet as net
import config as cfg
from layers.proposal_target_layer3 import proposal_target_layer
from layers.proposal_layer3 import proposal_layer
from layers.nms_wrapper import nms_detection
from utils.bbox_transform import clip_boxes

slim = tf.contrib.slim
pool = Pool(processes=4)

ckpt_dir = os.path.join(os.getcwd(), 'ckpt3')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


def build_targets(bbox_pred, iou_pred, gt_boxes, gt_cls, anchors, out_w, out_h, warmup):
    targets = pool.map(partial(proposal_target_layer,
                               anchors=anchors, out_w=out_w, out_h=out_h, warmup=warmup),
                       ((bbox_pred[i], iou_pred[i], gt_boxes[i], gt_cls[i])
                        for i in range(gt_boxes.shape[0])))  # multiprocessing

    num_boxes = np.sum(np.stack(t[0] for t in targets)).astype(np.float32)

    cls_target = np.stack(t[1] for t in targets)
    cls_mask = np.stack(t[2] for t in targets)
    iou_target = np.stack(t[3] for t in targets)
    iou_mask = np.stack(t[4] for t in targets)
    bbox_target = np.stack(t[5] for t in targets)
    bbox_mask = np.stack(t[6] for t in targets)

    return num_boxes, cls_target, cls_mask, iou_target, iou_mask, bbox_target, bbox_mask


class Network(object):
    def __init__(self, is_training=True, init_learning_rate=None):
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

        if is_training:  # training phase
            self.gt_boxes_ph = tf.placeholder(tf.float32)
            self.gt_cls_ph = tf.placeholder(tf.int8)
            self.warmup_ph = tf.placeholder(tf.bool)

            cls_predictions = []  # logits for cross-entropy, not predictions
            cls_targets = []
            cls_masks = []
            iou_predictions = []
            iou_targets = []
            iou_masks = []
            bbox_predictions = []
            bbox_targets = []
            bbox_masks = []
            total_boxes = 0

            for block in end_points:
                block_logits = end_points[block]
                # NHWC tensor, different in each block
                block_h, block_w = block_logits.get_shape()[1:3]
                block_logits = tf.reshape(
                    block_logits, shape=[-1, block_h*block_w, cfg.NUM_ANCHORS_CELL, 5 + cfg.NUM_CLASSES])

                # bbox predictions
                xy_pred = tf.sigmoid(block_logits[:, :, :, 0:2])
                wh_pred = tf.exp(block_logits[:, :, :, 2:4])
                bbox_pred = tf.concat([xy_pred, wh_pred], axis=-1)
                bbox_predictions.append(bbox_pred)

                # iou predictions
                iou_pred = tf.sigmoid(block_logits[:, :, :, 4:5])
                iou_predictions.append(iou_pred)

                # cls predictions, logits
                cls_pred = block_logits[:, :, :, 5:]
                cls_predictions.append(cls_pred)

                # TODO: faster here!!!
                num_boxes, cls_target, cls_mask, iou_target, iou_mask, bbox_target, bbox_mask = tf.py_func(build_targets,
                                                                                                           [bbox_pred, iou_pred,
                                                                                                            self.gt_boxes_ph, self.gt_cls_ph,
                                                                                                            self.anchors[
                                                                                                                block], block_w, block_h,
                                                                                                               self.warmup_ph],
                                                                                                           [tf.float32] * 7,
                                                                                                           name=block+'_proposal_target_layer')

                total_boxes += num_boxes
                cls_targets.append(cls_target)
                cls_masks.append(cls_mask)
                iou_targets.append(iou_target)
                iou_masks.append(iou_mask)
                bbox_targets.append(bbox_target)
                bbox_masks.append(bbox_mask)

            cls_predictions = tf.concat(cls_predictions, axis=1)
            cls_targets = tf.concat(cls_targets, axis=1)
            cls_masks = tf.concat(cls_masks, axis=1)
            iou_predictions = tf.concat(iou_predictions, axis=1)
            iou_targets = tf.concat(iou_targets, axis=1)
            iou_masks = tf.concat(iou_masks, axis=1)
            bbox_predictions = tf.concat(bbox_predictions, axis=1)
            bbox_targets = tf.concat(bbox_targets, axis=1)
            bbox_masks = tf.concat(bbox_masks, axis=1)

            rsum = tf.losses.Reduction.SUM

            cls_loss = tf.losses.softmax_cross_entropy(
                cls_targets, cls_predictions, cls_masks, scope='cls_loss', reduction=rsum) / total_boxes
            iou_loss = tf.losses.mean_squared_error(
                iou_targets*iou_masks, iou_predictions*iou_masks, scope='iou_loss', reduction=rsum) / total_boxes
            bbox_loss = tf.losses.mean_squared_error(
                bbox_targets*bbox_masks, bbox_predictions*bbox_masks, scope='bbox_loss', reduction=rsum) / total_boxes
            total_loss = cls_loss + iou_loss + bbox_loss

            self.global_step = tf.Variable(
                0, trainable=False, name='global_step')

            learning_rate = tf.train.exponential_decay(
                init_learning_rate, self.global_step, 12500, 0.9, staircase=True)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate).minimize(total_loss, self.global_step)

            # training summaries
            tf.summary.scalar('cls_loss', cls_loss)
            tf.summary.scalar('iou_loss', iou_loss)
            tf.summary.scalar('bbox_loss', bbox_loss)
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('learning_rate', learning_rate)

            self.merged = tf.summary.merge_all()

            self.writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), 'logs'), self.session.graph)

        else:  # inference phase
            self.box_coords = []
            self.box_cls = []
            self.box_scores = []

            for block in end_points:
                block_logits = end_points[block]
                # NHWC tensor, different in each block
                block_h, block_w = block_logits.get_shape()[1:3]
                block_logits = tf.reshape(
                    block_logits, shape=[-1, block_h*block_w, cfg.NUM_ANCHORS_CELL, 5 + cfg.NUM_CLASSES])

                # bbox predictions
                xy_pred = tf.sigmoid(block_logits[:, :, :, 0:2])
                wh_pred = tf.exp(block_logits[:, :, :, 2:4])
                bbox_pred = tf.concat([xy_pred, wh_pred], axis=-1)[0]

                # iou predictions
                iou_pred = tf.sigmoid(block_logits[:, :, :, 4:5])[0]

                # cls predictions
                cls_pred = tf.nn.softmax(block_logits[:, :, :, 5:])[0]

                # only 1 image per batch
                # keep top-n-score each block to apply nms to combined blocks
                box_pred, cls_inds, scores = tf.py_func(proposal_layer,
                                                        [bbox_pred, iou_pred, cls_pred,
                                                         self.anchors[block], block_w, block_h],
                                                        [tf.float32, tf.int8,
                                                            tf.float32],
                                                        name=block+'_proposal_layer')

                self.box_coords.append(box_pred)
                self.box_cls.append(cls_inds)
                self.box_scores.append(scores)

            self.box_coords = tf.concat(self.box_coords, axis=0)
            self.box_cls = tf.concat(self.box_cls, axis=0)
            self.box_scores = tf.concat(self.box_scores, axis=0)

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
                                                       self.warmup_ph: warmup})

        self.writer.add_summary(summary, step)

    def predict(self, image):
        image = np.expand_dims(image, axis=0)  # [1, H, W, C]

        # combined multi-layer bounding boxes
        box_coords, box_cls, box_scores = self.session.run([self.box_coords, self.box_cls, self.box_scores],
                                                           feed_dict={self.images_ph: image})

        keep = np.zeros(len(box_coords), dtype=np.int8)
        for i in cfg.NUM_CLASSES:
            inds = np.where(box_cls == i)[0]
            if len(inds) == 0:
                continue  # no i-objects in image

            dets = np.hstack([box_coords[inds], box_scores[inds]])

            keep_in_cls = nms_detection(np.ascontiguousarray(dets, dtype=np.float32),
                                        cfg.NMS_THRESH, use_gpu=cfg.USE_GPU)

            keep[inds[keep_in_cls]] = 1

        keep = np.where(keep > 0)[0]

        box_coords = box_coords[keep]
        box_cls = (box_cls[keep]).astype(np.int8)
        box_scores = box_scores[keep]

        # clip outside-region of boxes
        box_coords = clip_boxes(np.ascontiguousarray(box_coords, dtype=np.float32),
                                cfg.INP_SIZE, cfg.INP_SIZE)

    return box_coords, box_cls, box_scores


if __name__ == '__main__':
    Network(init_learning_rate=1e-3)  # training
