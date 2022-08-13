import time
import torch
from PIL import Image
from torchvision import transforms
from network_files import SparseRCNN, SparseRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2

def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = SparseRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



# 两个检测框框是否有交叉，根据结果选择要不要它
def bb_overlab(xmin1, ymin1, xmax1,ymax1, xmin2, ymin2, xmax2,ymax2):
    x1=xmin1
    y1=ymax1
    w1=xmax1-xmin1
    h1=ymax1-ymin1
    x2 = xmin2
    y2 = ymax2
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2

    if(x1>x2+w2):
        return 0,0,0,0
    if(y1>y2+h2):
        return 0,0,0,0
    if(x1+w1<x2):
        return 0,0,0,0
    if(y1+h1<y2):
        return 0,0,0,0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area/area1, overlap_area/area2,area1,area2



def main(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original_img = Image.open(path).convert('RGB')
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    #读入模型
    model = create_model(num_classes=13)
    train_weights = "pth/object_wenshi.pth"
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)
    model1 = create_model(num_classes=13)
    train_weights = "pth/object_ear.pth"
    model1.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model1.to(device)
    model2 = create_model(num_classes=13)
    train_weights = "pth/object_foot.pth"
    model2.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model2.to(device)

    model.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        box_delete=[]
        boxes1=[]
        for i in range(len(predict_boxes)-1):#剔除预测重复的框
            for j in range(i+1,len(predict_boxes)):
                a=predict_boxes[i]
                b=predict_boxes[j]
                c1,c2,area1,area2=bb_overlab(a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3])
                if c1>0.5 or c2>0.5:#IOU过大，认为是包围数据，则保留其中面积大的那一个
                    if area1<area2:
                        box_delete.append(i)
                    else:
                        box_delete.append(j)
        for i in range(len(predict_boxes)):
            if i not in box_delete:
                boxes1.append(predict_boxes[i])

    model1.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model1(init_img)
        predictions = model1(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        box_delete = []
        boxes2 = []
        for i in range(len(predict_boxes) - 1):  # 剔除预测重复的框
            for j in range(i + 1, len(predict_boxes)):
                a = predict_boxes[i]
                b = predict_boxes[j]
                c1, c2, area1, area2 = bb_overlab(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
                if c1 > 0.5 or c2 > 0.5:  # IOU过大，认为是包围数据，则保留其中面积大的那一个
                    if area1 < area2:
                        box_delete.append(i)
                    else:
                        box_delete.append(j)
        for i in range(len(predict_boxes)):
            if i not in box_delete:
                boxes2.append(predict_boxes[i])

    model2.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model2(init_img)
        predictions = model2(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        box_delete = []
        boxes3 = []
        for i in range(len(predict_boxes) - 1):  # 剔除预测重复的框
            for j in range(i + 1, len(predict_boxes)):
                a = predict_boxes[i]
                b = predict_boxes[j]
                c1, c2, area1, area2 = bb_overlab(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
                if c1 > 0.5 or c2 > 0.5:  # IOU过大，认为是包围数据，则保留其中面积大的那一个
                    if area1 < area2:
                        box_delete.append(i)
                    else:
                        box_delete.append(j)
        for i in range(len(predict_boxes)):
            if i not in box_delete:
                boxes3.append(predict_boxes[i])

    print(boxes1)#wenshi的预测框
    print(boxes2)#ear的预测框
    print(boxes3)#foot的预测框


if __name__ == '__main__':
    path='../data/2.png'
    main(path)

