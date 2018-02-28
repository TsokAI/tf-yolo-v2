# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_utils.py
# https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
from __future__ import absolute_import, division, print_function


def restore(sess, global_vars):
    print('from inception_v3 pretrained model')

    reader = NewCheckpointReader(os.path.join(
        os.getcwd(), 'model/inception_v3.ckpt'))

    # restore similars of global_vars and pretrained_vars, not include logits and global_step
    pretrained_var_names = [name + ':0'
                            for name in reader.get_variable_to_dtype_map().keys()
                            if not re.match('logits', name) and name != 'global_step']

    restoring_vars = [var for var in global_vars
                      if var.name in pretrained_var_names]

    restoring_var_names = [var.name[:-2] for var in restoring_vars]

    value_ph = tf.placeholder(dtype=tf.float32)

    for i in range(len(restoring_var_names)):
        sess.run(tf.assign(restoring_vars[i], value_ph),
                 feed_dict={value_ph: reader.get_tensor(restoring_var_names[i])})

    initializing_vars = [var for var in global_vars
                         if not var in restoring_vars]

    sess.run(tf.variables_initializer(initializing_vars))


def preprocess(images):
    # using keras preprocessing, not using distortion
    # rescale images to [-1, 1]
    return (images/255. - 0.5)*2.
