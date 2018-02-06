LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

NUM_CLASSES = len(LABEL_NAMES)

INP_SIZE = (416, 416)

IOU_THRESH = 0.6
CLS_SCALE = 1
OBJECT_SCALE = 5
NO_OBJECT_SCALE = 1
BBOX_SCALE = 1

PRE_NMS_KEEP_TOP_N = 1000
NMS_THRESH = 0.45

NUM_ANCHORS = 5
