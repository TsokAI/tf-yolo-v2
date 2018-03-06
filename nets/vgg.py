# https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
from __future__ import absolute_import, division, print_function
import sys
import os
import re
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

slim = tf.contrib.slim

endpoint = 'vgg_16'


def forward(inputs, num_outputs, is_training=True, scope=None):
    # modified (add batchnorm) from slim pretrained model
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = slim.repeat(inputs, 1, slim.conv2d, 64,
                                  [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                net = slim.repeat(net, 1, slim.conv2d, 128,
                                  [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')

                net = slim.repeat(net, 1, slim.conv2d, 256,
                                  [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                net = slim.repeat(net, 1, slim.conv2d, 512,
                                  [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')

                net = slim.repeat(net, 1, slim.conv2d, 512,
                                  [3, 3], scope='conv5')
                # net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # logits block
                net = slim.conv2d(net, num_outputs, [1, 1],
                                  activation_fn=None, normalizer_fn=None, scope='logits')

    return net


def restore(sess, global_vars):
    print('from vgg_16 pretrained model')

    reader = NewCheckpointReader(
        os.path.join(os.getcwd(), 'model/vgg_16.ckpt'))

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
        sys.stdout.write("\033[K")
        sess.run(tf.assign(restoring_vars[i], value_ph),
                 feed_dict={value_ph: reader.get_tensor(restoring_var_names[i])})

    print('restoring done')

    initializing_vars = [var for var in global_vars
                         if not var in restoring_vars]

    sess.run(tf.variables_initializer(initializing_vars))


def preprocess(images):
    # images: 4d tensor [batch_size, height, width, channels]
    # rgb_means subtraction on each image
    return tf.to_float(images) - [123.68, 116.78, 103.94]
