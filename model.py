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
from final_convnext_net import convnext_base
from shape_classification.final_zshape_convnext_net import convnext_base as convnext_base_shape
from model_predect import object_detection
from model_xml import find_bndbox_and_features

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
validation_preprocessing = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])

age_idx = {'商代早期': 0,
           '商代晚期': 1,
           '西周早期': 2,
           '西周中期': 3,
           '西周晚期': 4,
           '春秋中期': 5,
           '春秋早期': 6,
           '春秋晚期': 7,
           '战国中期': 8,
           '战国早期': 9,
           '战国晚期': 10
           }

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

def acc(list_num):
    total=0
    _, predicted = torch.max(list_num.data, 1)
    for i in list_num:
        total+=i

    return _/total

#输入一个list，返回一个装着前面list比例的list
def percent(list_num):
    len_list=len(list_num)
    sum=0
    for i in range(len_list):
        sum+=list_num[i]
    re_list=[]
    for i in range(len_list):
        re_list.append(round(list_num[i]/sum,2))
    return re_list

def test(image_path,model,model_shape):
    img = Image.open(image_path).convert('RGB')
    batch = validation_preprocessing(img)
    batch_t=torch.unsqueeze(batch, 0)
    outputs=model(batch_t)
    num4, predicted4 = torch.topk(outputs.data, k=4, dim=1)#选四个最大值，前值后索引

    "1.计算出最可能的四个年代及每个年代的概率（list_age和list_score）"
    list_score=percent(num4[0].tolist())#四个最大年代的概率
    list_age=[]#与上述概率对应的年代名称
    for i in range(len(predicted4[0])):
        for key, values in age_idx.items():
            predicted = float(predicted4[0][i])
            predicted = int(predicted)
            if values == predicted:
                break
        list_age.append(key)
    output = model_shape(batch_t)
    _, predicted = torch.max(output.data, 1)
    temp = int(_)
    if temp <= 5:  # 极大概率输入的图片不是青铜器
        print('输入的图片可能不是青铜器！')
        return {
            'status': -1,
            'message': '输入的图片可能不是青铜器！'
        }
    for i in range(len(list_age)):
        print('年代预测为',list_age[i],'的准确率为',list_score[i])

    "2.计算出用户输入图片的器形，即key"
    for key, values in species_idx.items():
            predicted = float(predicted)
            predicted = int(predicted)
            if values == predicted :
                break
    print('器形为：',key)#预测值就是key

    "3.依老师要求，每一种预测年代，都给出8张同器形的图片同年代的预测图片（理论上是4*8张，4代表预测了4个年代，有可能图片数量不够，那就有几张就显示几张）"
    "下述list都是4*8大小的，代表推荐的相似图片的读取路径和用于展示的信息。list的位置都是一一对应的"
    min_dis8_list=[]#代表距离，无意义
    min8_path_list=[]#代表图片的存储路径
    min8_xml_list = []  # 代表xml文件的存储路径
    min8_age_list=[]#代表年代，从此开始的信息都可以展示出来
    min8_shape_list=[]#代表器型
    min8_name_list=[]#代表器名
    min8_birth_list=[]#代表出土地
    min8_where_list=[]#代表现藏地

    for i in range(len(list_age)):
        min_dis8, min8_path,min8_xml, min8_age, min8_shape, min8_name, min8_birth, min8_where = model_find_similar_pictures.find_similar_pictures(outputs, list_age[i], key)
        min_dis8_list.append(min_dis8)
        min8_path_list.append(min8_path)
        min8_xml_list.append(min8_xml)
        min8_age_list.append(min8_age)
        min8_shape_list.append(min8_shape)
        min8_name_list.append(min8_name)
        min8_birth_list.append(min8_birth)
        min8_where_list.append(min8_where)
        break
    "举例，min8_path_list[0][0]代表概率最大的年代（list_age[0]、list_score[0]）推荐的第一张图片的路径，min8_path_list[2][5]代表概率第三大的年代（list_age[3]、list_score[3]）推荐的第六张图片的路径，"
    "min8_age_list/min8_shape_list/min8_name_list/min8_birth_list/min8_where_list代表这张图片可以显示出来的信息，[0][0]代表概率最大的年代推荐的第一张图片的各种信息"


    "4.给上述4*8张图片读入对应的xml文件，来添加上bbox"
    "根据min8_xml_list读取就行，所以和min8_path_list[0][0]相同"
    "可用find_bndbox_and_features函数读取xml文件中的bbox,读取出的bbox的格式为[xmin,ymin,xmax,ymax]"
    list_all_pos = []
    list_name = []
    for i in range(8):
        anno = find_bndbox_and_features(min8_xml_list[0][i])
        list_all_pos.append(anno[0])
        anno_name = [j.split('\t')[-1] for j in anno[1]]
        list_name.append(anno_name)
    print('标注bbox的位置为：',list_all_pos)
    print('标注bbox的名字为（暂时）',list_name)


    "5.将用户读入的图片，给出预测框，只需要预测框，不需要名字概率"
    "得到的预测bbox的格式为[xmin,ymin,xmax,ymax]"
    boxes=object_detection(img)
    # to list
    boxes = [i.tolist() for i in boxes]
    # 
    idx2age = dict([(v, k) for k, v in age_idx.items()])
    print(idx2age)
    min8_age_list[0] = [idx2age[int(float(i))] for i in min8_age_list[0]]
    print('预测框为：',boxes)
    return {"pred_score" : list_score, 
            "pred_year" : list_age, 
            "rec_pth" : min8_path_list[0],
            "rec_age" : min8_age_list[0],
            "rec_qixing" : min8_shape_list[0],
            "rec_name" : min8_name,
            "rec_bth" : min8_birth_list[0],
            "rec_place" : min8_where_list[0],
            "anno_box" : list_all_pos,
            "anno_name" : list_name, 
            "bbox" : boxes,
            "status": 200
            }

def inference(image_path):
    net_age = convnext_base(in_22k=True)
    trained_weight = torch.load("3_batch/final_net.pth",map_location='cpu')
    net_age.load_state_dict(trained_weight.state_dict())

    net_shape = convnext_base_shape(in_22k=True)
    trained_weight = torch.load("shape_classification/pth/final_shape.pth", map_location='cpu')
    net_shape.load_state_dict(trained_weight.state_dict())
    return test(image_path,net_age, net_shape)

if __name__ == '__main__':
    # for test
    # image_path = './static/images/0.jpeg'
    image_path='D:/python_projects/bronze_ware/data2/ori_images_png_2/1/01007.png'
    net_age = convnext_base(in_22k=True)
    trained_weight = torch.load("3_batch/final_net.pth",map_location='cpu')
    net_age.load_state_dict(trained_weight.state_dict())

    net_shape = convnext_base_shape(in_22k=True)
    trained_weight = torch.load("shape_classification/pth/final_shape.pth", map_location='cpu')
    net_shape.load_state_dict(trained_weight.state_dict())
    test(image_path,net_age, net_shape)