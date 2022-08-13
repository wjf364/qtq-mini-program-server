#本代码用来查找
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
    path_xml = os.path.join('static', path_xml)
    bndbox = [0, 0, 0, 0]
    domobj = xmldom.parse(path_xml)#固定语句1，用来读取path文件
    elementobj = domobj.documentElement#固定语句二
    sub_element_obj = elementobj.getElementsByTagName('name')#找名字
    sub_element_obj1 = elementobj.getElementsByTagName('bndbox')#找位置
    list_name=[]#用来存储一张图片中所有对应的名字
    list_all_pos=[]
    for i in range(len(sub_element_obj)):  # 用i来返回指定名字在xml中的索引
        name = sub_element_obj[i].firstChild.data.replace(' ', '_')#轻松找到名字
        list_name.append(name)
        if sub_element_obj1 is not None:
            list_temp = []
            bndbox[0] = int(sub_element_obj1[i].getElementsByTagName('xmin')[0].firstChild.data)
            list_temp.append(bndbox[0])
            bndbox[1] = int(sub_element_obj1[i].getElementsByTagName('ymin')[0].firstChild.data)
            list_temp.append(bndbox[1])
            bndbox[2] = int(sub_element_obj1[i].getElementsByTagName('xmax')[0].firstChild.data)
            list_temp.append(bndbox[2])
            bndbox[3] = int(sub_element_obj1[i].getElementsByTagName('ymax')[0].firstChild.data)
            list_temp.append(bndbox[3])
            list_all_pos.append(list_temp)
    return list_all_pos,list_name