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
from draw import draw_box

age_idx1 = {'商代早期': 0,
           '商代中期': 0,
           '商代晚期': 1,
           '西周早期': 2,
           '西周早期前段': 2.1,
           '西周早期后段': 2.2,
           '西周早期後段': 2.2,
           '西周中期': 3,
           '西周中期前段': 3.1,
           '西周中期後段': 3.2,
           '西周中期晚段': 3.2,
           '西周晚期': 4,
           '春秋中期': 5,
           '春秋早期': 6,
           '春秋晚期': 7,
           '战国中期': 8,
           '战国早期': 9,
           '战国晚期': 10,
           '戰國中期': 8,
           '戰國早期': 9,
           '戰國晚期': 10,
           '商代晚期或西周早期': 11
           }
def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = SparseRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=13)#12or68

    # load train weights
    train_weights = "pth/object_wenshi.pth"#要改
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    # category_index = v_setting.age_idx
    category_index = {v: k for k, v in age_idx1.items()}

    # load image
    # original_img = Image.open("D:/python_projects/bronze_ware/data2/ori_images_png_3/0001a.png").convert('RGB')
    original_img = Image.open("../data/3.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01507.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01516.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01688.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01689.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01690.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01693.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01777.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘圖_01778.png").convert('RGB')
    # original_img = Image.open("D:/python_projects/faste_rcnn/data/test_image/test/銘續_0120.png").convert('RGB')
    # from pil image to tensor, do not normalize image
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

        # t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        # t_end = time_synchronized()
        # print("inference+NMS time: {}".format(t_end - t_start))

        #提取出了预测的信息
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        print(len(predict_boxes[0]))

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.5,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        # 保存预测的图片结果
        # original_img.save("pth/test_result.jpg")


if __name__ == '__main__':
    main()

