# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
# https://www.tensorflow.org/tutorials/image_retraining#other_model_architectures
from __future__ import absolute_import, division, print_function
import sys
import os
import re
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

slim = tf.contrib.slim

endpoint = 'MobilenetV1'


def depthsep_conv2d(inputs, num_outputs, kernel, stride, scope=None):
    net = slim.separable_conv2d(inputs, None, kernel,
                                depth_multiplier=1,
                                stride=stride,
                                scope=scope + '_depthwise')

    net = slim.conv2d(net, num_outputs, [1, 1],
                      stride=1,
                      scope=scope + '_pointwise')

    return net


def forward(inputs, num_outputs, is_training=True, scope=None):
    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # root block
                net = slim.conv2d(inputs, 32, [3, 3],
                                  stride=2, scope='Conv2d_0')

                # separate blocks
                net = depthsep_conv2d(net, 64, [3, 3],
                                      stride=1, scope='Conv2d_1')
                net = depthsep_conv2d(net, 128, [3, 3],
                                      stride=2, scope='Conv2d_2')
                net = depthsep_conv2d(net, 128, [3, 3],
                                      stride=1, scope='Conv2d_3')
                net = depthsep_conv2d(net, 256, [3, 3],
                                      stride=2, scope='Conv2d_4')
                net = depthsep_conv2d(net, 256, [3, 3],
                                      stride=1, scope='Conv2d_5')
                net = depthsep_conv2d(net, 512, [3, 3],
                                      stride=2, scope='Conv2d_6')
                net = depthsep_conv2d(net, 512, [3, 3],
                                      stride=1, scope='Conv2d_7')
                net = depthsep_conv2d(net, 512, [3, 3],
                                      stride=1, scope='Conv2d_8')
                net = depthsep_conv2d(net, 512, [3, 3],
                                      stride=1, scope='Conv2d_9')
                net = depthsep_conv2d(net, 512, [3, 3],
                                      stride=1, scope='Conv2d_10')
                net = depthsep_conv2d(net, 512, [3, 3],
                                      stride=1, scope='Conv2d_11')
                net = depthsep_conv2d(net, 1024, [3, 3],
                                      stride=2, scope='Conv2d_12')
                net = depthsep_conv2d(net, 1024, [3, 3],
                                      stride=1, scope='Conv2d_13')

                # logits block
                net = slim.conv2d(net, num_outputs, [1, 1],
                                  activation_fn=None, normalizer_fn=None, scope='logits')

    return net


def restore(sess, global_vars):
    print('from MobilenetV1 pretrained model')
    reader = NewCheckpointReader(os.path.join(
        os.getcwd(), 'model/mobilenet_v1_1.0_224.ckpt'))

    # restore similars of global_vars and pretrained_vars, not include logits and global_step
    pretrained_var_names = [name + ':0'
                            for name in reader.get_variable_to_dtype_map().keys()
                            if not re.search('logits', name) and name != 'global_step']

    restoring_vars = [var for var in global_vars
                      if var.name in pretrained_var_names]

    restoring_var_names = [var.name[:-2] for var in restoring_vars]

    value_ph = tf.placeholder(dtype=tf.float32)

    for i in range(len(restoring_var_names)):
        print('loc:@' + restoring_var_names[i])
        sys.stdout.write("\033[F")
        sess.run(tf.assign(restoring_vars[i], value_ph),
                 feed_dict={value_ph: reader.get_tensor(restoring_var_names[i])})

    print()

    initializing_vars = [var for var in global_vars
                         if not var in restoring_vars]

    sess.run(tf.variables_initializer(initializing_vars))


def preprocess(images):
    # images: 4d tensor [batch_size, height, width, channels]
    return (images - 128.) / 128.
