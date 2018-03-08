DATASET = 'voc07'

LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

# DATASET = 'detrac'

# LABEL_NAMES = ['car', 'bus', 'van', 'others']

NUM_CLASSES = len(LABEL_NAMES)

INP_SIZE = 416  # rgb images

NUM_ANCHORS_CELL = 5

USE_GPU = True

# training
IOU_THRESH = 0.6  # positive rois

CLS_SCALE = 1
BBOX_SCALE = 1
OBJECT_SCALE = 5
NO_OBJECT_SCALE = 1

# inference
COEF_THRESH = 0.3  # detector confidence

PRE_NMS_TOP_N = 300  # keep top n before apply nms

NMS_THRESH = 0.45
