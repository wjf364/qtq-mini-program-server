#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/06/29 12:24:11
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   一些工具函数
'''
import base64
import os
import random
import time
from io import BytesIO

from PIL import Image


def img_to_base64(img_path):
    '''
    将本地图片转换为base64格式
    '''
    with open(img_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('ascii')
    return img_base64

def base64_to_img(img_base64, img_path):
    '''
    将base64格式的图片转换为本地图片
    '''
    bytes_data = base64.b64decode(img_base64)
    image = Image.open(BytesIO(bytes_data))
    # save
    image.save(img_path)
    return img_path

def get_random_file(path='static/images'):
    """
        获取随机图片
    """
    import os
    files = os.listdir(path)
    
    return os.path.join(path, random.choice(files))

def save_image(image, info="", path="static/data"):
    filename = info + str(time.time())
    filename = str(base64.urlsafe_b64encode(filename.encode("utf-8")), "utf-8") + ".jpg"
    # save image
    image_path = os.path.join(path, filename)
    image.save(image_path)
    return filename

if __name__ == '__main__':
    # img_base64 = img_to_base64('./static/images/0.jpeg')
    # # print(img_base64)
    # path = base64_to_img(img_base64, './static/images/new.jpeg')
    # print(path)
    im = Image.open('./static/images/0.jpeg')
    res = save_image(im)
    print(res)


