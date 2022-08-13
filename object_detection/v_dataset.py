import os
import numpy as np
import torch
import v_transforms
from v_dataloader import load_age_table1, load_age_table2, load_age_table3, BronzeWareDataset1,BronzeWareDataset2,BronzeWareDataset3

data_transform = {
        "train": v_transforms.Compose([v_transforms.ToTensor(),
                                     v_transforms.RandomHorizontalFlip(0.5)]),
        "test": v_transforms.Compose([v_transforms.ToTensor()])
    }
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
#第一批数据读入
data_path = '../data2'
ware_img_name, ware_age, ware_shape = load_age_table1(os.path.join(data_path, 'for_1.xlsx'))
for idx in range(len(ware_age)):
    if ware_age[idx][-1] == '\u3000':
        ware_age[idx] = ware_age[idx][:-1]
ware_age = [age_idx1[age] for age in ware_age]
ware_age = np.floor(np.asarray(ware_age))
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))
traindata1=ware_data

#第二批数据读入
#21
ware_img_name, ware_age, ware_shape = load_age_table2(os.path.join(data_path, 'for_2.1.xlsx'))
for idx in range(len(ware_age)):
    if ware_age[idx][-1] == '\u3000':
        ware_age[idx] = ware_age[idx][:-1]
ware_age = [age_idx1[age] for age in ware_age]
ware_age = np.floor(np.asarray(ware_age))
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))
traindata21=ware_data
#22
ware_img_name, ware_age, ware_shape = load_age_table2(os.path.join(data_path, 'for_2.2.xlsx'))
for idx in range(len(ware_age)):
    if ware_age[idx][-1] == '\u3000':
        ware_age[idx] = ware_age[idx][:-1]
ware_age = [age_idx1[age] for age in ware_age]
ware_age = np.floor(np.asarray(ware_age))
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))
traindata22=ware_data
#23
ware_img_name, ware_age, ware_shape = load_age_table2(os.path.join(data_path, 'for_2.3.xlsx'))
for idx in range(len(ware_age)):
    if ware_age[idx][-1] == '\u3000':
        ware_age[idx] = ware_age[idx][:-1]
ware_age = [age_idx1[age] for age in ware_age]
ware_age = np.floor(np.asarray(ware_age))
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))
traindata23=ware_data

#第三批数据读入
ware_img_name, ware_age, ware_shape = load_age_table3(os.path.join(data_path, 'for_3_object.xlsx'))
for idx in range(len(ware_age)):
    if ware_age[idx][-1] == '\u3000':
        ware_age[idx] = ware_age[idx][:-1]
    if ware_age[idx][-1] == '\xa0':
        ware_age[idx] = ware_age[idx][:-1]
ware_age = [age_idx1[age] for age in ware_age]
ware_age = np.floor(np.asarray(ware_age))
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))
traindata3=ware_data
#数据读入结束
# print(traindata1)
# print(traindata21)
# print(traindata3)

#开始整合数据
image_folder1 = os.path.join(data_path, 'ori_images_png')
image_folder21 = os.path.join(data_path, 'ori_images_png_2/1')
image_folder22 = os.path.join(data_path, 'ori_images_png_2/2')
image_folder23 = os.path.join(data_path, 'ori_images_png_2/3')
image_folder3 = os.path.join(data_path, 'ori_images_png_3')
mxl_folder1='../mark_completed_data/new_marked_data/1_batch'
mxl_folder21='../mark_completed_data/new_marked_data/2_batch/1'
mxl_folder22='../mark_completed_data/new_marked_data/2_batch/2'
mxl_folder23='../mark_completed_data/new_marked_data/2_batch/3'
mxl_folder3='../mark_completed_data/new_marked_data/3_batch'

trainset1 = BronzeWareDataset1(image_folder1,mxl_folder1, traindata1, transform=data_transform["train"])
trainset21 = BronzeWareDataset2(image_folder21,mxl_folder21, traindata21, transform=data_transform["train"])
trainset22 = BronzeWareDataset2(image_folder22,mxl_folder22, traindata22, transform=data_transform["train"])
trainset23 = BronzeWareDataset2(image_folder23,mxl_folder23, traindata23, transform=data_transform["train"])
trainset3 = BronzeWareDataset3(image_folder3,mxl_folder3, traindata3, transform=data_transform["train"])
# trainloader = torch.utils.data.DataLoader(trainset1+trainset21+trainset22+trainset23+trainset3, batch_size=64,shuffle=True, num_workers=0)
