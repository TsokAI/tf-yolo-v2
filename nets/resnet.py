# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
# resnet_v2_50 from slim
from __future__ import absolute_import, division, print_function
import os
import re
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
from nets.resnet_utils import subsample, conv2d_same

slim = tf.contrib.slim


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
                                   normalizer_fn=None, activation_fn=None, scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = conv2d_same(residual, depth_bottleneck, 3, stride=stride,
                               rate=rate, scope='conv2')

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

    return output


def resnet_v2_block(net, base_depth, num_units, stride, scope=None):
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
    with tf.variable_scope(scope, 'block', [net]):
        # unit scope is unit_%d/bottleneck_v2
        for i in range(num_units - 1):
            net = bottleneck(net, depth, base_depth, stride=1,
                             scope='unit_{}/bottleneck_v2'.format(i + 1))
        net = bottleneck(net, depth, base_depth, stride=stride,
                         scope='unit_{}/bottleneck_v2'.format(num_units))

    return net


def forward(inputs, num_outputs, is_training=True, scope=None):
    with tf.variable_scope(scope, 'resnet_v2_50', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # root_block
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None, normalizer_fn=None):
                net = conv2d_same(inputs, 64, 7, stride=2, scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

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

            net = slim.conv2d(net, num_outputs, [1, 1],
                              activation_fn=None, normalizer_fn=None,
                              scope='_logits_')

    return net


def restore(sess, global_vars):
    print('from resnet_v2_50 pretrained model')

    reader = NewCheckpointReader(os.path.join(
        os.getcwd(), 'model/resnet_v2_50.ckpt'))

    # restore both weights and biases from conv and shortcut layers
    restored_var_names = [name + ':0'
                          for name in reader.get_variable_to_dtype_map().keys()
                          if re.match('^.*weights$', name) or re.match('^.*biases$', name)]

    restored_vars = [var for var in global_vars
                     if var.name in restored_var_names]

    restored_var_names = [var.name[:-2] for var in restored_vars]

    value_ph = tf.placeholder(dtype=tf.float32)

    for i in range(len(restored_var_names)):
        sess.run(tf.assign(restored_vars[i], value_ph),
                 feed_dict={value_ph: reader.get_tensor(restored_var_names[i])})

    initialized_vars = [var for var in global_vars
                        if not var in restored_vars]

    sess.run(tf.variables_initializer(initialized_vars))


def preprocess(images):
    # images: 4d tensor [batch_size, height, width, channels]
    return images / 255.
