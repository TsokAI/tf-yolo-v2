# yolov2 with tensorflow

## it doesnt work well as i expected, i dont know why, please review code and let me know! thanks!

## dataset:
create symlink to 'data' from dataset location, 'data' folder contain 'images' and 'annotation' subfolder

tf-slim pretrained model in 'model' folder

annotation using pascal/voc xml format

using default yolov2's anchor in 416x416, can be scaled to different size

## training:
using numpy's axis (not pascal/voc's axis): (ymin,xmin) = (xmin,ymin) and (ymax,xmax) = (xmax,ymax)

network using VGG16 pretrained model (removed fc layers) from tf-slim with adding 2 conv layers (conv6, logits). vgg_16.ckpt put in model/

python3 train.py --num_epochs NUM_EPOCHS --batch_size NUM_IMAGES --learn_rate LEARN_RATE

losses collection (step, bbox, iou, class, total) will be saved in logs/losses_collection.txt

edit adamop to use AdamOptimizer instead of SGD with momentum and pretrained to use VGG16, MobilenetV1, Resnet_v2_50 model from tf-slim instead of initialization from scratch

## validation - testing:

## demo:

## todo:
evaluate model

fast bbox_transform in network

fast postprocess with nms
