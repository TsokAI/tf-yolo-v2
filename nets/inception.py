# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_utils.py
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
from __future__ import absolute_import, division, print_function
import sys
import os
import re
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

slim = tf.contrib.slim


def depth(d, min_depth=16, depth_mul=1.0):
    return max(int(d * depth_mul), min_depth)


def forward(inputs, is_training=True, scope=None):
    with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # regular blocks
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='VALID'):

                    net = slim.conv2d(inputs, depth(
                        32), [3, 3], stride=2, scope='Conv2d_1a_3x3')

                    net = slim.conv2d(net, depth(
                        32), [3, 3], scope='Conv2d_2a_3x3')

                    net = slim.conv2d(net, depth(
                        64), [3, 3], padding='SAME', scope='Conv2d_2b_3x3')

                    net = slim.max_pool2d(
                        net, [3, 3], stride=2, scope='MaxPool_3a_3x3')

                    net = slim.conv2d(net, depth(
                        80), [1, 1], scope='Conv2d_3b_1x1')

                    net = slim.conv2d(net, depth(
                        192), [3, 3], scope='Conv2d_4a_3x3')

                    net = slim.max_pool2d(
                        net, [3, 3], stride=2, scope='MaxPool_5a_3x3')

                # inception blocks
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):

                    with tf.variable_scope('Mixed_5b'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                64), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                48), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                                   scope='Conv2d_0b_5x5')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                64), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                   scope='Conv2d_0b_3x3')
                            branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                   scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_5c'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                64), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                48), [1, 1], scope='Conv2d_0b_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                                   scope='Conv_1_0c_5x5')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                                   scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                   scope='Conv2d_0b_3x3')
                            branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                   scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_5d'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                64), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                48), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                                   scope='Conv2d_0b_5x5')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                64), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                   scope='Conv2d_0b_3x3')
                            branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                   scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_6a'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                                   padding='VALID', scope='Conv2d_1a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                64), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                                   scope='Conv2d_0b_3x3')
                            branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                                   padding='VALID', scope='Conv2d_1a_1x1')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                       scope='MaxPool_1a_3x3')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2])

                    with tf.variable_scope('Mixed_6b'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                128), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                                   scope='Conv2d_0b_1x7')
                            branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                   scope='Conv2d_0c_7x1')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                128), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                                   scope='Conv2d_0b_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                                   scope='Conv2d_0c_1x7')
                            branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                                   scope='Conv2d_0d_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                   scope='Conv2d_0e_1x7')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_6c'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                160), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                                   scope='Conv2d_0b_1x7')
                            branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                   scope='Conv2d_0c_7x1')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                160), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                   scope='Conv2d_0b_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                                   scope='Conv2d_0c_1x7')
                            branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                   scope='Conv2d_0d_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                   scope='Conv2d_0e_1x7')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_6d'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                160), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                                   scope='Conv2d_0b_1x7')
                            branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                   scope='Conv2d_0c_7x1')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                160), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                   scope='Conv2d_0b_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                                   scope='Conv2d_0c_1x7')
                            branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                   scope='Conv2d_0d_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                   scope='Conv2d_0e_1x7')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_6e'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                                   scope='Conv2d_0b_1x7')
                            branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                   scope='Conv2d_0c_7x1')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                                   scope='Conv2d_0b_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                   scope='Conv2d_0c_1x7')
                            branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                                   scope='Conv2d_0d_7x1')
                            branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                   scope='Conv2d_0e_1x7')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                   scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_7a'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                            branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                192), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                                   scope='Conv2d_0b_1x7')
                            branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                   scope='Conv2d_0c_7x1')
                            branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                       scope='MaxPool_1a_3x3')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2])

                    with tf.variable_scope('Mixed_7b'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                320), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                384), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = tf.concat(axis=3, values=[
                                slim.conv2d(branch_1, depth(384), [
                                            1, 3], scope='Conv2d_0b_1x3'),
                                slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                448), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(
                                branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                            branch_2 = tf.concat(axis=3, values=[
                                slim.conv2d(branch_2, depth(384), [
                                            1, 3], scope='Conv2d_0c_1x3'),
                                slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

                    with tf.variable_scope('Mixed_7c'):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, depth(
                                320), [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, depth(
                                384), [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = tf.concat(axis=3, values=[
                                slim.conv2d(branch_1, depth(384), [
                                            1, 3], scope='Conv2d_0b_1x3'),
                                slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, depth(
                                448), [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(
                                branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                            branch_2 = tf.concat(axis=3, values=[
                                slim.conv2d(branch_2, depth(384), [
                                            1, 3], scope='Conv2d_0c_1x3'),
                                slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.avg_pool2d(
                                net, [3, 3], scope='AvgPool_0a_3x3')
                            branch_3 = slim.conv2d(
                                branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                        net = tf.concat(
                            axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    return net


def restore(sess, global_vars):
    print('from inception_v3 pretrained model')

    reader = NewCheckpointReader(os.path.join(
        os.getcwd(), 'model/inception_v3.ckpt'))

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
    # using keras preprocessing, using color distortion
    # rescale images to [-1, 1]
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    if is_training:
        images = tf.map_fn(preprocess_for_train, images)

    images = tf.multiply(tf.subtract(images, 0.5), 2.0)

    return images
