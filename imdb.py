from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2
import xml.etree.cElementTree as ctree
import config as cfg

label2cls = {}
for idx, label in enumerate(cfg.LABEL_NAMES):
    label2cls[label] = idx


def load_image(anno_dir, images_dir, xml):
    # image in shape of [height, width, num_channels]
    # boxes in shape of [num_gt_boxes, (xmin, ymin, xmax, ymax)], scaled in inp_size
    # classes in shape of [num_gt_boxes, (cls)]
    # skip truncated, difficult in xml
    root = ctree.parse(os.path.join(anno_dir, xml)).getroot()
    image_name = root.find('filename').text
    image_size = root.find('size')
    image_height = float(image_size.find('height').text)
    image_width = float(image_size.find('width').text)
    classes = []
    boxes = []
    for box in root.findall('object'):
        classes.append(label2cls[box.find('name').text])
        bndbox = box.find('bndbox')
        # using numpy's axis (not pascal/voc)
        # X-----------> y
        # |
        # |
        # V x
        boxes.append([float(bndbox.find('ymin').text),
                      float(bndbox.find('xmin').text),
                      float(bndbox.find('ymax').text),
                      float(bndbox.find('xmax').text)])

    # scale box coords to target size
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, 0::2] *= cfg.INP_SIZE / image_height
    boxes[:, 1::2] *= cfg.INP_SIZE / image_width

    classes = np.array(classes, dtype=np.int8)

    image = cv2.imread(os.path.join(images_dir, image_name))
    # image preprocessing
    # cv2 using BGR channels for imread, convert to RGB and normalize
    image = cv2.resize(image, (cfg.INP_SIZE, cfg.INP_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if np.random.randint(0, 2):  # randomly left-right flipping
        image = cv2.flip(image, 1)
        boxes[:, 1::2] = cfg.INP_SIZE - boxes[:, 1::2]

    return image, boxes, classes


class Imdb:
    def __init__(self, anno_dir, images_dir, batch_size, max_images=None):
        self.anno_dir = anno_dir
        self.images_dir = images_dir

        self.anno = os.listdir(self.anno_dir)
        if max_images is not None:
            self.anno = self.anno[:max_images]

        self.size = len(self.anno)
        self.batch_size = batch_size
        self.start_idx = 0

    def get_size(self):
        return self.size

    def next_batch(self):
        np.random.shuffle(self.anno)

        while True:
            batch_images = []
            batch_boxes = []
            batch_classes = []

            end_idx = min(self.start_idx + self.batch_size, self.size)
            for xml in self.anno[self.start_idx:end_idx]:
                image, boxes, classes = load_image(
                    self.anno_dir, self.images_dir, xml)
                batch_images.append(image)
                batch_boxes.append(boxes)
                batch_classes.append(classes)

            batch_images = np.asarray(batch_images, dtype=np.float32)

            # add padding, list np.ndarray -> tf.tensor
            num_images = batch_images.shape[0]
            max_boxes_im = max([len(clss) for clss in batch_classes])

            batch_boxes_pad = np.zeros(
                (num_images, max_boxes_im, 4), dtype=np.float32)
            batch_classes_pad = np.full(
                (num_images, max_boxes_im), -1, dtype=np.int8)  # -1 mean dontcare, removed in compute targets

            for i in range(num_images):
                num_boxes_im = len(batch_classes[i])
                batch_boxes_pad[i, 0:num_boxes_im, :] = batch_boxes[i]
                batch_classes_pad[i, 0:num_boxes_im] = batch_classes[i]

            yield batch_images, batch_boxes_pad, batch_classes_pad

            # delete whatever yielded
            del batch_images
            del batch_boxes
            del batch_boxes_pad
            del batch_classes
            del batch_classes_pad

            self.start_idx = end_idx if end_idx < self.size else 0

            # complete epoch
            if self.start_idx == 0:
                break
