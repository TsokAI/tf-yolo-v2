from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import config as cfg
from py_compute_targets import compute_targets_batch
from utils.bbox import bbox_transform
from model_wrapper import get_model

slim = tf.contrib.slim

model, forward = get_model(name=cfg.model)

ckpt_dir = os.path.join(cfg.workspace, model['save_dir'])
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


class Network:
    def __init__(self, session, im_shape, is_training=True, lr=1e-3, adamop=False, pretrained=False):
        self.sess = session
        self.is_training = is_training

        # network's placeholders
        self.images_ph = tf.placeholder(
            tf.float32, shape=[None, im_shape[0], im_shape[1], 3])

        self.anchors_ph = tf.placeholder(
            tf.float32, shape=[cfg.num_anchors, 2])

        logits = forward(self.images_ph,
                         num_outputs=cfg.num_anchors * (cfg.num_classes + 5),
                         scope=model['endpoint'])

        ft_shape = logits.get_shape().as_list()[1:3]

        logits = tf.reshape(logits,
                            [-1, ft_shape[0] * ft_shape[1], cfg.num_anchors, cfg.num_classes + 5])

        bbox_pred = tf.concat([tf.sigmoid(logits[:, :, :, 0:2]), tf.exp(logits[:, :, :, 2:4])],
                              axis=3)

        self.box_pred = tf.py_func(bbox_transform,
                                   [bbox_pred, self.anchors_ph,
                                       ft_shape[0], ft_shape[1]],
                                   tf.float32, name='box_pred')

        self.iou_pred = tf.sigmoid(logits[:, :, :, 4:5])

        self.cls_pred = tf.nn.softmax(logits[:, :, :, 5:])

        if self.is_training:
            # network's placeholders in training
            self.boxes_ph = tf.placeholder(tf.float32, shape=None)

            self.classes_ph = tf.placeholder(tf.int8, shape=None)

            self.num_boxes_batch_ph = tf.placeholder(tf.float32, shape=None)

            _cls, _cls_mask, _iou, _iou_mask, _bbox, _bbox_mask = tf.py_func(compute_targets_batch,
                                                                             [im_shape, ft_shape,
                                                                              self.box_pred, self.iou_pred,
                                                                              self.boxes_ph, self.classes_ph, self.anchors_ph],
                                                                             [tf.float32] * 6, name='targets')

            # network's losses, focal loss on cls?
            self.bbox_loss = tf.losses.mean_squared_error(labels=_bbox * _bbox_mask,
                                                          predictions=bbox_pred * _bbox_mask,
                                                          reduction=Reduction.SUM) / self.num_boxes_batch_ph
            self.iou_loss = tf.losses.mean_squared_error(labels=_iou * _iou_mask,
                                                         predictions=self.iou_pred * _iou_mask,
                                                         reduction=Reduction.SUM) / self.num_boxes_batch_ph
            self.cls_loss = tf.losses.mean_squared_error(labels=_cls * _cls_mask,
                                                         predictions=self.cls_pred * _cls_mask,
                                                         reduction=Reduction.SUM) / self.num_boxes_batch_ph

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

        self.saver = tf.train.Saver(max_to_keep=3)

        self.load_ckpt(pretrained)

    def load_ckpt(self, pretrained):
        # restore model with ckpt/pretrain or init
        try:
            print('trying to restore last checkpoint')
            last_ckpt_path = tf.train.latest_checkpoint(
                checkpoint_dir=ckpt_dir)
            self.saver.restore(self.sess, save_path=last_ckpt_path)
            print('restored checkpoint from:', last_ckpt_path)
        except:
            if self.is_training:
                print('init variables')
                restored_vars = []
                global_vars = tf.global_variables()

                if pretrained:  # restore from tf-slim model
                    if os.path.exists(os.path.join(cfg.workspace, model['valid_path'])):
                        print('from ' + model['endpoint'])

                        import re
                        from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

                        reader = NewCheckpointReader(
                            os.path.join(cfg.workspace, model['ckpt_path']))

                        restored_var_names = [name + ':0'
                                              for name in reader.get_variable_to_dtype_map().keys()
                                              if re.match(model['restore'], name)]

                        restored_vars = [var for var in global_vars
                                         if var.name in restored_var_names]

                        restored_var_names = [var.name[:-2]
                                              for var in restored_vars]

                        value_ph = tf.placeholder(tf.float32, shape=None)
                        for i in range(len(restored_var_names)):
                            self.sess.run(tf.assign(restored_vars[i], value_ph),
                                          feed_dict={value_ph: reader.get_tensor(restored_var_names[i])})

                initialized_vars = list(set(global_vars) - set(restored_vars))
                self.sess.run(tf.variables_initializer(initialized_vars))

    def train(self, batch_images, batch_boxes, batch_classes, anchors, num_boxes_batch):
        assert self.is_training

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
                        save_path=os.path.join(ckpt_dir, model['endpoint']),
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
