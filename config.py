# DATASET = 'voc'
DATASET = 'detrac'

# LABEL_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#                'dog', 'horse', 'motorbike', 'person', 'pottedplant',
#                'sheep', 'sofa', 'train', 'tvmonitor']

LABEL_NAMES = ['car', 'bus', 'van', 'others']
# {'car': 175379, 'bus': 10691, 'van': 19207, 'others': 1524} -> 206801 examples
# weight_class = log(alpha*total_examples / class_examples), alpha=1

NUM_CLASSES = len(LABEL_NAMES)

INP_SIZE = 352  # rgb images

NUM_ANCHORS_CELL = 3

USE_GPU = True

# training
IOU_THRESH = 0.5  # ignored rois threshold

# CLASS_SCALE = [0.165, 2.962, 2.376, 4.91]
CLASS_SCALE = [1] * NUM_CLASSES
COORD_SCALE = 1
OBJECT_SCALE = 5
NO_OBJECT_SCALE = 1

# inference
COEF_THRESH = 0.3  # detector confidence

PRE_NMS_TOP_N = 200  # keep top n before apply nms

NMS_THRESH = 0.3
