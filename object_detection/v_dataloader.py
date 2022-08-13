#有关于数据导入的各种函数
#不训练所有的标注框，仅训练在v_setting.features_idx_all中的标注框
import glob
import os
import pandas as pd
from skimage import io, transform, color
from lxml import etree
import numpy as np
import v_transforms

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from v_setting import features_foot,features_ear,features_line
#上面的用于控制目标检测检测哪部分

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

#整合，三批均不同
class BronzeWareDataset1(Dataset):#feature可以被out了
    def __init__(self, root_dir,root_dir2, ware_data, transform=None):#root_dir为图片的位置，root_dir2为对应xml存储的位置
        self.root_dir = root_dir
        self.annotations_root=root_dir2
        self.ware_img_name = ware_data[:, 0]
        self.ware_age = ware_data[:, 1]
        self.ware_shape=ware_data[:,2]
        self.transform = transform
        self.ware_img = []
        self.ware_xml=[]

        for img_name in self.ware_img_name:#根据excel里的文件名字来找对应的图片，而不是根据图片来找名字
            img_name0 = img_name[:2] + '_' + img_name[2:] + '.png' # '.jpg'
            img_name1 = img_name[:2] + '_' + img_name[2:] + '.xml'
            img_name = os.path.join(self.root_dir, img_name0)
            img_name1 = os.path.join(self.annotations_root, img_name1)
            image = Image.open(img_name).convert('RGB')
            self.ware_img.append(image)#读入图片了
            self.ware_xml.append(img_name1)#读入对应的xml文件
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xml_path=self.ware_xml[idx]#打开xml文件
        with open(xml_path,encoding='utf-8') as fid:#打开上述对应的xml文件
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:#注意，是for循环，把一张图片里的多个都读入进来了
            if obj["name"] in features_foot:#只有在features_idx_all才读入进来
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                #注意，这个地方之后做目标检测的时候要改进，label目前用整张图片的label
                # print(self.ware_age[idx])
                a=float(self.ware_age[idx])
                a=int(a)
                labels.append(a)
                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
        if len(boxes)==0:
            for obj in data["object"]:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                # 注意，这个地方之后做目标检测的时候要改进，label目前用整张图片的label
                # print(self.ware_age[idx])
                a = float(self.ware_age[idx])
                a = int(a)
                labels.append(a)
                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
                break
        image = self.ware_img[idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))#不同之处
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            a = float(self.ware_age[idx])
            a = int(a)
            labels.append(a)
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
class BronzeWareDataset2(Dataset):#feature可以被out了
    def __init__(self, root_dir,root_dir2, ware_data, transform=None):#root_dir为图片的位置，root_dir2为对应xml存储的位置
        self.root_dir = root_dir
        self.annotations_root=root_dir2
        self.ware_img_name = ware_data[:, 0]
        self.ware_age = ware_data[:, 1]
        self.ware_shape=ware_data[:,2]
        self.transform = transform
        self.ware_img = []
        self.ware_xml=[]

        for img_name in self.ware_img_name:#根据excel里的文件名字来找对应的图片，而不是根据图片来找名字
            img_name0 = img_name + '.png' # '.jpg'
            img_name1 = img_name + '.xml'
            img_name = os.path.join(self.root_dir, img_name0)
            img_name1 = os.path.join(self.annotations_root, img_name1)
            image = Image.open(img_name).convert('RGB')
            self.ware_img.append(image)#读入图片了
            self.ware_xml.append(img_name1)#读入对应的xml文件
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xml_path=self.ware_xml[idx]#打开xml文件
        with open(xml_path,encoding='utf-8') as fid:#打开上述对应的xml文件
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:#注意，是for循环，把一张图片里的多个都读入进来了
            if obj["name"] in features_foot:#只有在features_idx_all才读入进来
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                #注意，这个地方之后做目标检测的时候要改进，label目前用整张图片的label
                # print(self.ware_age[idx])
                a=float(self.ware_age[idx])
                a=int(a)
                labels.append(a)
                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
        image = self.ware_img[idx]
        if len(boxes) == 0:
            for obj in data["object"]:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                # 注意，这个地方之后做目标检测的时候要改进，label目前用整张图片的label
                # print(self.ware_age[idx])
                a = float(self.ware_age[idx])
                a = int(a)
                labels.append(a)
                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
                break
        # print(xml_path,boxes,labels,iscrowd)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))#不同之处
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # print(xml_path, boxes, labels, iscrowd)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)

        # return image, torch.from_numpy(np.asarray([self.ware_age[idx]]).astype('float64')),torch.from_numpy(np.asarray([self.ware_shape[idx]]).astype('float64'))
        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            a = float(self.ware_age[idx])
            a = int(a)
            labels.append(a)
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
class BronzeWareDataset3(Dataset):#feature可以被out了
    def __init__(self, root_dir,root_dir2, ware_data, transform=None):#root_dir为图片的位置，root_dir2为对应xml存储的位置
        self.root_dir = root_dir
        self.annotations_root=root_dir2
        self.ware_img_name = ware_data[:, 0]
        self.ware_age = ware_data[:, 1]
        self.ware_shape=ware_data[:,2]
        self.transform = transform
        self.ware_img = []
        self.ware_xml=[]

        for img_name in self.ware_img_name:#根据excel里的文件名字来找对应的图片，而不是根据图片来找名字
            img_name0 = img_name + '.png' # '.jpg'
            img_name1 = img_name + '.xml'
            img_name = os.path.join(self.root_dir, img_name0)
            img_name1 = os.path.join(self.annotations_root, img_name1)
            image = Image.open(img_name).convert('RGB')
            self.ware_img.append(image)#读入图片了
            self.ware_xml.append(img_name1)#读入对应的xml文件
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xml_path=self.ware_xml[idx]#打开xml文件
        with open(xml_path,encoding='utf-8') as fid:#打开上述对应的xml文件
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:#注意，是for循环，把一张图片里的多个都读入进来了
            if obj["name"] in features_foot:#只有在features_idx_all才读入进来
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                #注意，这个地方之后做目标检测的时候要改进，label目前用整张图片的label
                # print(self.ware_age[idx])
                a=float(self.ware_age[idx])
                a=int(a)
                labels.append(a)
                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
        image = self.ware_img[idx]
        if len(boxes) == 0:
            for obj in data["object"]:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                boxes.append([xmin, ymin, xmax, ymax])
                # 注意，这个地方之后做目标检测的时候要改进，label目前用整张图片的label
                # print(self.ware_age[idx])
                a = float(self.ware_age[idx])
                a = int(a)
                labels.append(a)
                if "difficult" in obj:
                    iscrowd.append(int(obj["difficult"]))
                else:
                    iscrowd.append(0)
                break
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))#不同之处
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            image, target = self.transform(image, target)

        # return image, torch.from_numpy(np.asarray([self.ware_age[idx]]).astype('float64')),torch.from_numpy(np.asarray([self.ware_shape[idx]]).astype('float64'))
        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            a = float(self.ware_age[idx])
            a = int(a)
            labels.append(a)
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))









