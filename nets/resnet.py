# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
from __future__ import absolute_import, division, print_function
import sys
import os
import re
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
from nets.resnet_utils import conv2d_same, resnet_v2_block

slim = tf.contrib.slim


# modified for yolo3, not compatible with yolo2
def forward(inputs, num_outputs, is_training=True, scope=None):
    end_points = {}

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=slim.batch_norm):
        # resnet50 backbone
        with tf.variable_scope(scope, 'resnet_v2_50', [inputs]):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # root block
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(inputs, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                # residual blocks
                net, _ = resnet_v2_block(
                    net, base_depth=64, num_units=3, stride=2, scope='block1')

                net, block2_aux = resnet_v2_block(
                    net, base_depth=128, num_units=4, stride=2, scope='block2')

                net, block3_aux = resnet_v2_block(
                    net, base_depth=256, num_units=4, stride=2, scope='block3')

                net, _ = resnet_v2_block(
                    net, base_depth=512, num_units=2, stride=1, scope='block4')
                net = slim.batch_norm(
                    net, activation_fn=tf.nn.relu, scope='postnorm')

        # feature pyramid, residual blocks, convolution_transpose
        # aux applied batchnorm
        with tf.variable_scope('block4_deconv'):
            end_points['block4'] = slim.conv2d(
                net, num_outputs, [1, 1], normalizer_fn=None, activation_fn=None, scope='conv1')

        with tf.variable_scope('block3_deconv'):
            block3_c = block3_aux.get_shape()[-1]

            block3_dec = slim.conv2d(net, block3_c, [1, 1], scope='conv1')

            block3_dec = slim.conv2d_transpose(
                block3_dec, block3_c, [3, 3], stride=2, scope='conv2')

            block3_dec = block3_dec + block3_aux

            end_points['block3'] = slim.conv2d(block3_dec, num_outputs, [
                1, 1], normalizer_fn=None, activation_fn=None, scope='conv3')

        with tf.variable_scope('block2_deconv'):
            block2_c = block2_aux.get_shape()[-1]

            block2_dec = slim.conv2d(block3_dec, block2_c, [
                                     1, 1], scope='conv1')

            block2_dec = slim.conv2d_transpose(
                block2_dec, block2_c, [3, 3], stride=2, scope='conv2')

            block2_dec = block2_dec + block2_aux

            end_points['block2'] = slim.conv2d(block2_dec, num_outputs, [
                1, 1], normalizer_fn=None, activation_fn=None, scope='conv3')

    return end_points


def restore(sess, global_vars):
    print('from resnet_v2_50 pretrained model')

    reader = NewCheckpointReader(os.path.join(
        os.getcwd(), 'model/resnet_v2_50.ckpt'))

    # restore similars of global_vars and pretrained_vars, not include logits and global_step
    pretrained_var_names = [name + ':0'
                            for name in reader.get_variable_to_dtype_map().keys()
                            if name != 'global_step']

    restoring_vars = [var for var in global_vars
                      if var.name in pretrained_var_names]

    restoring_var_names = [var.name[:-2] for var in restoring_vars]

    value_ph = tf.placeholder(dtype=tf.float32)

    for i in range(len(restoring_var_names)):
        print('loc:@' + restoring_var_names[i])
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        sess.run(tf.assign(restoring_vars[i], value_ph),
                 feed_dict={value_ph: reader.get_tensor(restoring_var_names[i])})

    print('restoring done')

    initializing_vars = [var for var in global_vars
                         if not var in restoring_vars]

    sess.run(tf.variables_initializer(initializing_vars))


def preprocess_for_train(image):
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image


def preprocess(images, is_training=True):
    # rescale images to [-1, 1]
    # resnet_v2 using inception preprocess
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    if is_training:
        images = tf.map_fn(preprocess_for_train, images)

    images = tf.multiply(tf.subtract(images, 0.5), 2.0)

    return images
