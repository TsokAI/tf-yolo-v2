# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py
# atrous convolution is dilated convolution
# no definition for resnet 18 and 34 layers because of no pretrained models in tf-slim
from __future__ import absolute_import, division, print_function
import tensorflow as tf

slim = tf.contrib.slim


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The subsampling factor.
        scope: Optional variable_scope.
    Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels] with the
          input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride=1, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    Note that
        net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
        net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
        net = subsample(net, factor=stride)
    whereas
        net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
    Args:
        inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
        num_outputs: An integer, the number of output filters.
        kernel_size: An int with the kernel_size of the filters.
        stride: An integer, the output stride.
        rate: An integer, rate for atrous convolution.
        scope: Scope.
    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                           padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', scope=scope)
