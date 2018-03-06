from __future__ import absolute_import, division, print_function
import os
import time
from datetime import timedelta
from imdb import Imdb
from network import Network

data_dir = os.path.join(os.getcwd(), 'data')
anno_dir = os.path.join(data_dir, 'warmup')
images_dir = os.path.join(data_dir, 'images')


train_params = {
    'epochs': 150,
    'warmup_epochs': 5,
    'batch_size': 12,
    'warmup_batch_size': 4,
    'lr': 1e-5,
    'warmup_lr': 1e-4
}

train_t = 0
step = 0

print('start training')

imdb = Imdb(anno_dir, images_dir,
            batch_size=train_params['warmup_batch_size'])

net = Network(is_training=True, learning_rate=train_params['warmup_lr'])

# warmup epochs
for epoch in range(1, train_params['warmup_epochs'] + 1):
    epoch_t = time.time()

    for images, gt_boxes, gt_cls in imdb.next_batch():
        step, bbox_loss, iou_loss, cls_loss = net.fit(images, gt_boxes, gt_cls)

        if step % 1 == 0:
            print('step: {0} - bbox_loss: {1} - iou_loss: {2} - cls_loss: {3}'.format(
                step, bbox_loss, iou_loss, cls_loss))

    epoch_t_dif = time.time() - epoch_t
    train_t += epoch_t_dif

    print('epoch: {0} - time: {1}'.format(epoch,
                                          str(timedelta(seconds=epoch_t_dif))))

net.save_ckpt(0)

imdb.set_batch_size(batch_size=train_params['batch_size'])
net.refresh_opt(learning_rate=train_params['lr'])

# training epochs
for epoch in range(1, train_params['epochs'] + 1):
    epoch_t = time.time()

    for images, gt_boxes, gt_cls in imdb.next_batch():
        step, bbox_loss, iou_loss, cls_loss = net.fit(images, gt_boxes, gt_cls)

        if step % 1 == 0:
            print('step: {0} - bbox_loss: {1} - iou_loss: {2} - cls_loss: {3}'.format(
                step, bbox_loss, iou_loss, cls_loss))

    epoch_t_dif = time.time() - epoch_t
    train_t += epoch_t_dif

    print('epoch: {0} - time: {1}'.format(epoch,
                                          str(timedelta(seconds=epoch_t_dif))))

    if epoch % 10 == 0:
        net.save_ckpt(step)

print('training done - time: {}'.format(str(timedelta(seconds=train_t))))
