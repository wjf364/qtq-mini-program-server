#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/06/30 12:22:22
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   目标检测模型
'''


import torch
from torch import nn
from torch.nn import functional as F
from torchvision.io.image import read_image
from torchvision.io import ImageReadMode
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import nms
import logging

MODEL_SINGLETON = None

def build_model(model_name = 'retinanet_resnet50_fpn', weights=None):
    if model_name == 'retinanet_resnet50_fpn':
        model = retinanet_resnet50_fpn(weights = weights)
        model = model.cuda()
        model.eval()
        return model
    else:
        raise NotImplementedError('model not found')


def object_detection(image_path):
    # init model
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
    model = build_model(weights = weights)
    # preprocess
    img = None
    if image_path.endswith('.jpg'):
        img = read_image(image_path)
    elif image_path.endswith('.png'):
        img = read_image(image_path, ImageReadMode.RGB)
    else:
        img = read_image(image_path, ImageReadMode.RGB)
    preprocess = weights.transforms()
    batch = [preprocess(img).cuda()]
    # inference
    prediction = model(batch)[0]
    # postprocess
    #  nms
    inds = nms(prediction["boxes"], prediction["scores"], iou_threshold=0.3)
    prediction["labels"] = prediction["labels"][inds]
    prediction["boxes"] = prediction["boxes"][inds,:]
    prediction["scores"] = prediction["scores"][inds]
    #  filter
    inds = prediction["scores"] > 0.2
    prediction["labels"] = prediction["labels"][inds]
    prediction["boxes"] = prediction["boxes"][inds,:]
    prediction["scores"] = prediction["scores"][inds]
    
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    
    # draw boxes
    # box = draw_bounding_boxes(img, boxes=prediction["boxes"],
    #                         labels=labels,
    #                         colors="red",
    #                         width=4, font_size=60)
    
    
    prediction["labels"] = prediction["labels"].detach().cpu().numpy().tolist()
    prediction["boxes"] = prediction["boxes"].detach().cpu().numpy().tolist()
    prediction["scores"] = prediction["scores"].detach().cpu().numpy().tolist()
    
    
    return prediction, labels, img

if __name__ == '__main__':
    # for test
    image_path = './static/images/0.jpeg'
    prediction, labels, img = object_detection(image_path)
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=60)
    
    im = to_pil_image(box.detach())
    im.show()


