from __future__ import absolute_import, division, print_function
import argparse
import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
import config as cfg
from network import Network
from blob import BlobLoader

slim = tf.contrib.slim

data_dir = os.path.join(os.getcwd(), 'data')
anno_dir = os.path.join(data_dir, 'VOCtrainval_0712a')
images_dir = os.path.join(data_dir, 'VOCtrainval_0712i')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learn_rate', type=float, default=1e-3)
args = parser.parse_args()

print('num_epochs: {0} - batch_size: {1} - learn_rate: {2}'.format(
    args.num_epochs, args.batch_size, args.learn_rate))

# tf configuration
xla = tf.ConfigProto()
xla.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

net = Network(session=tf.Session(config=xla), im_shape=cfg.inp_size, is_training=True,
              lr=args.learn_rate, adamop=True, pretrained=True)
