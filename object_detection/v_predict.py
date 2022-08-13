import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms

import v_setting
from network_files import SparseRCNN, SparseRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
# from draw_box_utils import draw_box
import cv2
import numpy as np
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocessing = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    # normalize
])

def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = SparseRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

#本函数，输入图片的路径(不需要输入路径那么麻烦了，至今输入img就可以，img的格式是Image.open(path).convert('RGB')，符合对图片做的处理)，即可把裁剪后的图片返回，暂时定义返回五张
def depart_image_position(ori_path):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    path = ori_path
    original_img = Image.open(path).convert('RGB')

    # create model
    model = create_model(num_classes=13)

    # load train weights
    # train_weights = "/home/stu/wjf/resNetFpn19_all_type.pth"#要改
    train_weights = "pth/object_foot.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)


    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        predictions = model(img.to(device))[0]

        #提取出了预测的信息,预测信息本身就是排好序的，我暂时选取前五个最大的，而且只需要box信息即可，如果小于五个，那就不断重复，直到满足五个
        predict_boxes = predictions["boxes"].to("cpu").numpy()

        box_save=[]
        for i in range(len(predict_boxes)):
            box_save.append(predict_boxes[i])
            if i>5:
                break

        if len(box_save)>0:
            j=0
            while len(box_save)<5:
                box_save.append(box_save[j])  # 先将p_变成list形式进行拼接，注意输入为一个tuple:
                j=j+1

            boxes=box_save[0:5]

            img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)#转变格式
            Image.fromarray(img).show()
            img_part=[]
            for k in range(len(boxes)):
                    left_h = int(boxes[k][1])
                    left_w = int(boxes[k][0])
                    right_h = int(boxes[k][3])
                    right_w = int(boxes[k][2])
                    sub_img = img[left_h: right_h, left_w: right_w]
                    # print(sub_img,type(sub_img))
                    # part=Image.fromarray(sub_img).show()
                    sub_img = Image.fromarray(sub_img)
                    img_part.append(sub_img)

        else:#竟然还有预测框为0的情况，那就用图片本身好了,完全忘记了这个条件了，疏忽疏忽
            img_part = []
            img1 = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)  # 转变格式
            sub_img1 = Image.fromarray(img1)
            img_part.append(sub_img1)
            while len(img_part)<5:
                img_part.append(sub_img1)  # 先将p_变成list形式进行拼接，注意输入为一个tuple:
        print(img_part)
        return img_part


if __name__ == '__main__':
    #注意，这里面有问题，原来图片经过变换了，所以还是需要未经变换的图片才行，所以还是需要path
    #或者不需要path，在找对应图片的时候存储一下原图片就好了
    path="D:/python_projects/bronze_ware/data2/ori_images_png_3/0001a.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_00107.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_00516.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01507.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01516.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01688.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01689.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01690.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01693.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01777.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_01778.png"
    # path = "D:/python_projects/faste_rcnn/data/ori_images_png/銘圖_0120.png"

    boxes=depart_image_position(path)