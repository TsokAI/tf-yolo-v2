from __future__ import absolute_import, division, print_function
import os
import time
from datetime import timedelta
from imdb import Imdb
from network import Network

data_dir = os.path.join(os.getcwd(), 'data')
anno_dir = os.path.join(data_dir, 'annotation')
images_dir = os.path.join(data_dir, 'images')

train_params = {
    'epochs': 100,
    'warmup_epochs': 10,
    'batch_size': 16,
    'lr': 1e-6
}

train_t = 0

imdb = Imdb(anno_dir, images_dir,
            batch_size=train_params['batch_size'])

net = Network(is_training=True,
              learning_rate=train_params['lr'])

print('start training')

# warmup epochs
for epoch in range(train_params['warmup_epochs']):
    epoch_t = time.time()

    for images, gt_boxes, gt_cls in imdb.next_batch():
        net.fit(images, gt_boxes, gt_cls, warmup=True)

    epoch_t_dif = time.time() - epoch_t
    train_t += epoch_t_dif

net.save_ckpt()

# training epochs
# for epoch in range(1, train_params['epochs'] + 1):
#     epoch_t = time.time()

#     for images, gt_boxes, gt_cls in imdb.next_batch():
#         net.fit(images, gt_boxes, gt_cls)

#     epoch_t_dif = time.time() - epoch_t
#     train_t += epoch_t_dif

#     print('epoch: {0} - time: {1}'.format(epoch,
#                                           str(timedelta(seconds=epoch_t_dif))))

#     if epoch % 10 == 0:
#         net.save_ckpt()

# print('training done - time: {}'.format(str(timedelta(seconds=train_t))))
