from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2
import xml.etree.cElementTree as ctree
import config as cfg

label2cls = {}
for idx, label in enumerate(cfg.label_names):
    label2cls[label] = idx


def prep_image(anno_dir, xml):
    # images in shape of [height, width, num_channels]
    # bboxes in shape of [num_gt_boxes, (xmin, ymin, xmax, ymax)], scaled in cfg.inp_size
    # bclasses in shape of [num_gt_boxes, (cls)]
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
        # opposite to pascal/voc annotation
        boxes.append([float(bndbox.find('ymin').text),
                      float(bndbox.find('xmin').text),
                      float(bndbox.find('ymax').text),
                      float(bndbox.find('xmax').text)])

    # scale box coords to target size
    boxes = np.array(boxes, dtype=np.float32) * float(cfg.inp_size)
    boxes[0::2] /= image_height
    boxes[1::2] /= image_width

    classes = np.array(classes, dtype=np.float32)

    image = cv2.imread(os.path.join(cfg.data_dir, 'images', image_name))
    image = cv2.resize(image, (cfg.inp_size, cfg.inp_size)) / 255.0

    return image, boxes, classes


class BlobLoader:
    def __init__(self, anno_dir, batch_size=1):
        self.anno_dir = anno_dir
        self.anno = os.listdir(os.path.join(cfg.data_dir, anno_dir))
        self.num_anno = len(self.anno)
        self.batch_size = batch_size

    def next_batch(self):
        batch_images = []
        batch_boxes = []
        batch_classes = []
        while True:
            for xml in self.anno[np.random.choice(self.num_anno, self.batch_size, replace=False)]:
                image, boxes, classes = prep_image(self.anno_dir, xml)
                batch_images.append(image)
                batch_boxes.append(boxes)
                batch_classes.append(classes)

            yield batch_images, batch_boxes, batch_classes

            del batch_images
            del batch_boxes
            del batch_classes
            batch_images = []
            batch_boxes = []
            batch_classes = []
