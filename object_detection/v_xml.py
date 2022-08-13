#本代码用来返回一张图片上所有标注框的名字，用来处理数据集，找到符合要求的（标记腿的）
import glob
import xml.dom.minidom as xmldom
from PIL import Image
import os
import cv2
import numpy as np
# import z_setting
import random
import tqdm
from xml.etree import ElementTree as ET
#第一步，输出一张图片上的所有feature以及对应的位置

def find_bndbox_and_features(path_xml):#用来找一张图片的全部feature及对应的位置
    bndbox = [0, 0, 0, 0]
    domobj = xmldom.parse(path_xml)#固定语句1，用来读取path文件
    elementobj = domobj.documentElement#固定语句二
    sub_element_obj = elementobj.getElementsByTagName('name')#找名字，名字就对应er zu 和 wenshi
    list_name=[]#用来存储一张图片中所有对应的名字
    for i in range(len(sub_element_obj)):  # 用i来返回指定名字在xml中的索引
        name = sub_element_obj[i].firstChild.data.replace(' ', '_')#轻松找到名字
        list_name.append(name)
    return list_name#找到一个xml文件中的所有名字了

#
# path_xml='mark_completed_data/marked_data/mix_xml/1b 纹饰_00001.xml'
# # #
# b=find_bndbox_and_features(path_xml)
# print(b)
# print(a[0:2])