from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import config as cfg
from py_compute_targets import compute_targets_batch
from utils.bbox import bbox_transform

slim = tf.contrib.slim

_SUM = tf.losses.Reduction.SUM

# backbone convolution network definition
if cfg.model == 'vgg':  # using vgg_16
    from nets.vgg import model, forward, restore, preprocess
elif cfg.model == 'resnet':  # using resnet_v2_50
    from nets.resnet import model, forward, restore, preprocess
elif cfg.model == 'mobilenet':  # using MobilenetV1
    from nets.mobilenet import model, forward, restore, preprocess
else:  # add new model in nets, skip restore, pretrained is False
    raise Exception('invalid model')


class Network:  # computation graph
    def __init__(self, session, im_shape, is_training=True, lr=1e-3, adamop=True, pretrained=False):
        # tensorflow session
        self.sess = session

        # network's placeholders
        self.images_ph = tf.placeholder(
            tf.float32, shape=[None, im_shape[0], im_shape[1], 3])

        self.anchors_ph = tf.placeholder(
            tf.float32, shape=[cfg.num_anchors, 2])

        preprocessed_images = preprocess(self.images_ph)

        logits = forward(preprocessed_images,
                         num_outputs=cfg.num_anchors * (cfg.num_classes + 5),
                         is_training=is_training,
                         scope=model)

        ft_shape = logits.get_shape().as_list()[1:3]

        logits = tf.reshape(logits,
                            [-1, ft_shape[0] * ft_shape[1], cfg.num_anchors, cfg.num_classes + 5])

        bbox_pred = tf.concat([tf.sigmoid(logits[:, :, :, 0:2]), tf.exp(logits[:, :, :, 2:4])],
                              axis=3)

        # slow operation, parallel with openmp
        self.box_pred = tf.py_func(bbox_transform,
                                   [bbox_pred, self.anchors_ph,
                                       ft_shape[0], ft_shape[1]],
                                   tf.float32, name='box_pred')

        self.iou_pred = tf.sigmoid(logits[:, :, :, 4:5])

        self.cls_pred = tf.nn.softmax(logits[:, :, :, 5:])

        if is_training:
            # network's placeholders in training
            self.boxes_ph = tf.placeholder(tf.float32, shape=None)

            self.classes_ph = tf.placeholder(tf.int8, shape=None)

            self.num_boxes_batch_ph = tf.placeholder(tf.float32, shape=None)

            _cls, _cls_mask, _iou, _iou_mask, _bbox, _bbox_mask = tf.py_func(compute_targets_batch,
                                                                             [im_shape, ft_shape,
                                                                              self.box_pred, self.iou_pred,
                                                                              self.boxes_ph, self.classes_ph, self.anchors_ph],
                                                                             [tf.float32] * 6, name='targets')

            # network's losses, cross-entropy loss on cls?
            # losses normalized with number of groundtruth boxes
            self.bbox_loss = tf.losses.mean_squared_error(labels=_bbox * _bbox_mask,
                                                          predictions=bbox_pred * _bbox_mask,
                                                          reduction=_SUM) / self.num_boxes_batch_ph
            self.iou_loss = tf.losses.mean_squared_error(labels=_iou * _iou_mask,
                                                         predictions=self.iou_pred * _iou_mask,
                                                         reduction=_SUM) / self.num_boxes_batch_ph
            self.cls_loss = tf.losses.mean_squared_error(labels=_cls * _cls_mask,
                                                         predictions=self.cls_pred * _cls_mask,
                                                         reduction=_SUM) / self.num_boxes_batch_ph

            # joint training with sum of losses
            self.total_loss = self.bbox_loss + self.iou_loss + self.cls_loss

            # network's optimizer
            self.global_step = tf.Variable(
                initial_value=0, trainable=False, name='global_step')

            if adamop:  # using Adam, better with small batch_size
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                    loss=self.total_loss, global_step=self.global_step)
            else:  # using SGD_momentum + nesterov
                # poor performance with small batch_size (<=10)
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True).minimize(
                    loss=self.total_loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=1)

        self.load_ckpt_or_init(pretrained)

    def load_ckpt_or_init(self, pretrained=False):
        # restore model with ckpt/pretrain or init
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=ckpt_dir)
            self.saver.restore(self.sess, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            print('init variables')
            if pretrained:  # using slim pretrained model
                # restore weights, biases and batchnorm
                restore(self.sess, tf.global_variables())
            else:  # xavier random init
                self.sess.run(tf.global_variables_initializer())

    def train(self, batch_images, batch_boxes, batch_classes, anchors, num_boxes_batch):
        step, bbox_loss, iou_loss, cls_loss, _ = self.sess.run([self.global_step,
                                                                self.bbox_loss, self.iou_loss, self.cls_loss,
                                                                self.optimizer],
                                                               feed_dict={self.images_ph: batch_images,
                                                                          self.boxes_ph: batch_boxes,
                                                                          self.classes_ph: batch_classes,
                                                                          self.anchors_ph: anchors,
                                                                          self.num_boxes_batch_ph: num_boxes_batch})

        return step, bbox_loss, iou_loss, cls_loss

    def save_ckpt(self, step):
        self.saver.save(self.sess,
                        save_path=os.path.join(ckpt_dir, model),
                        global_step=self.global_step)

        print('saved checkpoint at step {}'.format(step))

    def predict(self, scaled_images, anchors):
        box_pred, iou_pred, cls_pred = self.sess.run([self.box_pred, self.iou_pred, self.cls_pred],
                                                     feed_dict={self.images_ph: scaled_images,
                                                                self.anchors_ph: anchors})

        return box_pred, iou_pred, cls_pred


if __name__ == '__main__':
    Network(tf.Session(), im_shape=cfg.inp_size,
            pretrained=True)  # trying to init network
