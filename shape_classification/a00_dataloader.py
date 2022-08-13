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


def load_age_table(file_path):
    age_table = pd.read_excel(file_path)#读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_age = np.asarray(age_table.iloc[:, 2])
    ware_img_name = np.asarray(age_table.iloc[:, 3])
    ware_remark = np.asarray(age_table.iloc[:, 4])
    ware_shape = np.asarray(age_table.iloc[:, 5])

    return ware_name, ware_age, ware_img_name, ware_shape


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

        return image, torch.from_numpy(np.asarray([self.ware_age[idx]]).astype('float64')),torch.from_numpy(np.asarray([self.ware_shape[idx]]).astype('float64'))


if __name__ == '__main__':
    data_path = '../bronze_ware'

    # age_table_file = os.path.join(data_path, 'age.xlsx')
    # _, ware_age, ware_img_name, _ = load_age_table(age_table_file)
    #
    # for idx in range(ware_img_name.size):
    #     ware_img_name[idx] = ware_img_name[idx][:2] + '_' + ware_img_name[idx][2:] + '.jpg'
    #
    # show_image(os.path.join(data_path, 'images', ware_img_name[0]), ware_age[0])

    dataset = BronzeWareDataset(data_path)

    fig = plt.figure()

    for i in range(len(dataset)):
        image, age = dataset[i]

        print(i, image.shape, age)

        # ax = plt.subplot(1, 4, i + 1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        # show_landmarks(**sample)
        #
        # if i == 3:
        #     plt.show()
        #     break








