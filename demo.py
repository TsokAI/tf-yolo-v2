from __future__ import absolute_import, division, print_function
import os
import time
from datetime import timedelta
import numpy as np
import cv2
import tensorflow as tf
import config as cfg
from network import Network

slim = tf.contrib.slim

xla = tf.ConfigProto()
xla.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=xla),
              im_shape=cfg.INP_SIZE, is_training=False)
