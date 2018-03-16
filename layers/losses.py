from __future__ import absolute_import, division, print_function
import tensorflow as tf


def softmax_focal(onehot_labels, logits, weights=1.0, alpha=0.25, gamma=2.0, scope=None):
    with tf.name_scope(scope, 'softmax_focal_loss'):
        # CE = sigma(-ti*log(pi)) for softmax
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=onehot_labels, logits=logits)

        predictions = tf.nn.softmax(logits)

        p_t = tf.reduce_sum(onehot_labels*predictions, axis=-1)

        focal_weights = alpha*tf.pow(1 - p_t, gamma)

        focal_loss = focal_weights*cross_entropy*weights

        focal_loss = tf.reduce_sum(focal_loss)

    return focal_loss


def sigmoid_focal(onehot_labels, logits, weights=1.0, alpha=0.25, gamma=2.0, scope=None):
    with tf.name_scope(scope, 'sigmoid_focal_loss'):
        # CE = sigma(-ti*log(pi) - (1-ti)*log(1-pi)) for sigmoid
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=onehot_labels, logits=logits)

        predictions = tf.sigmoid(logits)

        gamma_weights = tf.pow(
            1 - (onehot_labels*predictions + (1 - onehot_labels)*(1 - predictions)), gamma)

        alpha_weights = onehot_labels*alpha + (1 - onehot_labels)*(1 - alpha)

        focal_loss = tf.reduce_sum(
            alpha_weights*gamma_weights*cross_entropy, axis=-1)*weights

        focal_loss = tf.reduce_sum(focal_loss)

    return focal_loss
