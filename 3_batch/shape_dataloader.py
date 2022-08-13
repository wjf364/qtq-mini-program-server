#有关于数据导入的各种函数
import os
import pandas as pd
from skimage import io, transform, color

import numpy as np

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#三批数据的顺序都不一样，所以每个读入的顺序都要修改
def load_age_table1(file_path):
    age_table = pd.read_excel(file_path,dtype=str)#读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_age = np.asarray(age_table.iloc[:, 2])
    ware_img_name = np.asarray(age_table.iloc[:, 3])
    ware_remark = np.asarray(age_table.iloc[:, 4])
    ware_shape = np.asarray(age_table.iloc[:, 5])

    return ware_img_name, ware_age, ware_shape

def load_age_table2(file_path):
    age_table = pd.read_excel(file_path,dtype=str)  # 读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_age = np.asarray(age_table.iloc[:, 2])
    ware_book = np.asarray(age_table.iloc[:, 3])
    ware_shape = np.asarray(age_table.iloc[:, 4])
    ware_where = np.asarray(age_table.iloc[:, 5])
    born_place = np.asarray(age_table.iloc[:, 6])

    return ware_id, ware_age, ware_shape

def load_age_table3(file_path):
    age_table = pd.read_excel(file_path,dtype=str)  # 读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_name_num = np.asarray(age_table.iloc[:, 2])
    ware_age = np.asarray(age_table.iloc[:, 3])
    ware_shape = np.asarray(age_table.iloc[:, 4])
    ware_where = np.asarray(age_table.iloc[:, 5])
    born_place = np.asarray(age_table.iloc[:, 6])

    return ware_id, ware_age, ware_shape


#第一批数据与第二第三批数据的命名规则不同，所以分别用三种BronzeWareDataset
class BronzeWareDataset(Dataset):#给随机分配数据集（名字）关联上对应图片的函数
    def __init__(self, root_dir, ware_data, transform=None):
        self.root_dir = root_dir
        self.ware_img_name = ware_data[:, 0]
        self.ware_age = ware_data[:, 1]
        self.ware_shape=ware_data[:,2]
        self.transform = transform
        self.ware_img = []
        for img_name in self.ware_img_name:#根据excel里的文件名字来找对应的图片，而不是根据图片来找名字
            #上面注释的原因，excel里的文件名字全属于图片（在图片里全能找到对应），但是反之不能
            img_name = img_name[:2] + '_' + img_name[2:] + '.png' # '.jpg'
            img_name = os.path.join(self.root_dir, img_name)
            image = Image.open(img_name).convert('RGB')

            self.ware_img.append(image)
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.transform(self.ware_img[idx])

        return image, torch.from_numpy(np.asarray([self.ware_age[idx]]).astype('float64'))


class new_BronzeWareDataset(Dataset):#给随机分配数据集（名字）关联上对应图片的函数
    def __init__(self, root_dir, ware_data, transform=None):
        self.root_dir = root_dir
        self.ware_img_name = ware_data[:, 0]
        self.ware_age = ware_data[:, 1]
        self.ware_shape=ware_data[:,2]
        self.transform = transform
        self.ware_img = []
        for img_name in self.ware_img_name:#根据excel里的文件名字来找对应的图片，而不是根据图片来找名字
            #上面注释的原因，excel里的文件名字全属于图片（在图片里全能找到对应），但是反之不能
            img_name = img_name + '.png' # '.jpg'
            img_name = os.path.join(self.root_dir, img_name)
            image = Image.open(img_name).convert('RGB')
            self.ware_img.append(image)
        # print(type(self.ware_age))

    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.transform(self.ware_img[idx])

        return image, torch.from_numpy(np.asarray([self.ware_age[idx]]).astype('float64'))


class new3_BronzeWareDataset(Dataset):#给随机分配数据集（名字）关联上对应图片的函数
    def __init__(self, root_dir, ware_data, transform=None):
        self.root_dir = root_dir
        self.ware_img_name = ware_data[:, 0]
        self.ware_age = ware_data[:, 1]
        self.ware_shape=ware_data[:,2]
        self.transform = transform
        self.ware_img = []
        for img_name in self.ware_img_name:#根据excel里的文件名字来找对应的图片，而不是根据图片来找名字
            #上面注释的原因，excel里的文件名字全属于图片（在图片里全能找到对应），但是反之不能
            img_name = img_name + '.png' # '.jpg'
            img_name = os.path.join(self.root_dir, img_name)
            image = Image.open(img_name).convert('RGB')
            self.ware_img.append(image)
        # print(type(self.ware_age))

    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.transform(self.ware_img[idx])

        return image, torch.from_numpy(np.asarray([self.ware_age[idx]]).astype('float64'))