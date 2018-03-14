# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
from __future__ import absolute_import, division, print_function
import sys
import os
import re
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
from nets.resnet_utils import subsample, conv2d_same

slim = tf.contrib.slim


@ slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.
    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    scope: Optional variable_scope.
    Returns:
        The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]):
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(
            inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = conv2d_same(residual, depth_bottleneck, 3, stride=stride,
                               rate=rate, scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

    return output


def resnet_v2_block(inputs, base_depth, num_units, stride, scope=None):
    """Helper function for creating a resnet_v2 block.
    Args:
        net: A tensor of size [batch, height, width, channels].
        depth: The depth of layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
          All other units have stride=1.
        scope: The scope of the block.
    Returns:
        A resnet_v2 block.
    """
    depth = 4 * base_depth
    with tf.variable_scope(scope, 'block', [inputs]):
        net = inputs
        # unit scope is unit_%d/bottleneck_v2
        for i in range(num_units - 1):
            net = bottleneck(net, depth, base_depth, stride=1,
                             scope='unit_{}/bottleneck_v2'.format(i + 1))
        net = bottleneck(net, depth, base_depth, stride=stride,
                         scope='unit_{}/bottleneck_v2'.format(num_units))

    return net


def forward(inputs, is_training=True, scope=None):
    with tf.variable_scope(scope, 'resnet_v2_50', [inputs]):
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # root block
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(inputs, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

                # residual blocks
                net = resnet_v2_block(
                    net, base_depth=64, num_units=3, stride=2, scope='block1')
                net = resnet_v2_block(
                    net, base_depth=128, num_units=4, stride=2, scope='block2')
                net = resnet_v2_block(
                    net, base_depth=256, num_units=6, stride=2, scope='block3')
                net = resnet_v2_block(
                    net, base_depth=512, num_units=3, stride=1, scope='block4')
                net = slim.batch_norm(
                    net, activation_fn=tf.nn.relu, scope='postnorm')

    return net


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


def preprocess(images):
    # rescale images to [-1, 1]
    # resnet_v2 using inception preprocess (keras, no distortion)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    images = tf.multiply(tf.subtract(images, 0.5), 2.0)

    return images
