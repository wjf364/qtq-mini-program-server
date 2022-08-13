#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/06/30 12:22:22
@Author  :   Tang Chuan
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   目标检测模型
'''
import glob
import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import model_find_similar_pictures
from final_zshape_convnext_net import convnext_base

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
validation_preprocessing = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])

species_idx={
    '矮扁球腹鼎':0,
    '早期獸首蹄足圓鼎':1,
    '垂腹方鼎':2,
    '越式鼎B':3,
    '淺鼓腹鼎':4,
    '晚期獸首蹄足圓鼎':5,
    '晚期獸首蹄足鼎':6,
    '扁足圓鼎':7,
    '錐足方鼎':8,
    '異形鼎':9,
    '圓鼎':10,
    '高蹄足圓鼎':11,
    '尖錐足圓鼎':12,
    '半球形腹圓鼎':13,
    '越式鼎A':14,
    '超半球腹或半球腹鼎':15,
    '扁足方鼎':16,
    '柱足方鼎':17,
    '小口鼎':18,
    '无':19,
    '收腹圓鼎':20,
    '匜鼎':21,
    '蹄足方鼎':22,
    '罐鼎':23,
    '半球腹或超半球腹圓鼎':24,
    '圓錐形足圓鼎':25,
    '鬲鼎':26,
    '半球腹形圓鼎':27,
    '垂腹圓鼎':28,
    '束腰平底鼎':29
}


def test(image_path,model):
    img = Image.open(image_path).convert('RGB')
    batch = validation_preprocessing(img)
    batch_t=torch.unsqueeze(batch, 0)
    outputs=model(batch_t)
    _, predicted = torch.max(outputs.data, 1)

    for key, values in species_idx.items():
            predicted = float(predicted)
            predicted = int(predicted)
            if values == predicted :
                break
    print(key)#预测值就是key
    pre = key




if __name__ == '__main__':
    image_path='../data/3.png'
    net = convnext_base(in_22k=True)
    trained_weight = torch.load("pth/final_shape.pth",map_location='cpu')
    net.load_state_dict(trained_weight.state_dict())
    test(image_path, net)