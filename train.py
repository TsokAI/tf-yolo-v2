from __future__ import absolute_import, division, print_function
import os
import argparse
import time
from datetime import timedelta
from imdb import Imdb
from network import Network

data_dir = os.path.join(os.getcwd(), 'data')
anno_dir = os.path.join(data_dir, 'annotation')
images_dir = os.path.join(data_dir, 'images')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learn_rate', type=float, default=1e-3)
args = parser.parse_args()

imdb = Imdb(anno_dir, images_dir,
            batch_size=args.batch_size,
            max_images=500)

net = Network(is_training=True, lr=args.learn_rate)

train_t = 0
step = 0

print('start training')

for epoch in range(1, args.num_epochs + 1):
    epoch_t = time.time()

    for images, gt_boxes, gt_cls in imdb.next_batch():
        step, bbox_loss, iou_loss, cls_loss = net.fit(images, gt_boxes, gt_cls)

        if step % 100 == 0:
            print('step: {0} - bbox_loss: {1} - iou_loss: {2} - cls_loss: {3}'.format(
                step, bbox_loss, iou_loss, cls_loss))

    epoch_t_dif = time.time() - epoch_t
    train_t += epoch_t_dif

    print('epoch: {0} - time: {1}'.format(epoch,
                                          str(timedelta(seconds=epoch_t_dif))))

    if epoch % 5 == 0:
        net.save_ckpt(step)

print('training done - time: {}'.format(str(timedelta(seconds=train_t))))
