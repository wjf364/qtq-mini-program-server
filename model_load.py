import os
import pandas as pd
from skimage import io, transform, color

import numpy as np

#三批数据的顺序都不一样，所以每个读入的顺序都要修改
def load_age_table1(file_path):
    age_table = pd.read_excel(file_path,dtype=str)#读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_age = np.asarray(age_table.iloc[:, 2])
    ware_img_name = np.asarray(age_table.iloc[:, 3])
    ware_remark = np.asarray(age_table.iloc[:, 4])
    ware_shape = np.asarray(age_table.iloc[:, 5])
    ware_location = np.asarray(age_table.iloc[:, 6])
    ware_birth = np.asarray(age_table.iloc[:, 7])

    return ware_img_name, ware_age, ware_shape,ware_name,ware_location,ware_birth#图片命名、年代、器型

def load_age_table2(file_path):
    age_table = pd.read_excel(file_path,dtype=str)  # 读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_age = np.asarray(age_table.iloc[:, 2])
    ware_book = np.asarray(age_table.iloc[:, 3])
    ware_shape = np.asarray(age_table.iloc[:, 4])
    ware_where = np.asarray(age_table.iloc[:, 5])
    born_place = np.asarray(age_table.iloc[:, 6])

    return ware_id, ware_age, ware_shape,ware_name,ware_where,born_place#图片命名、年代、器型

def load_age_table3(file_path):
    age_table = pd.read_excel(file_path,dtype=str)  # 读入age.xlsx

    ware_id = np.asarray(age_table.iloc[:, 0])
    ware_name = np.asarray(age_table.iloc[:, 1])
    ware_name_num = np.asarray(age_table.iloc[:, 2])
    ware_age = np.asarray(age_table.iloc[:, 3])
    ware_shape = np.asarray(age_table.iloc[:, 4])
    ware_where = np.asarray(age_table.iloc[:, 5])
    born_place = np.asarray(age_table.iloc[:, 6])

    return ware_id, ware_age, ware_name#图片命名、年代、器型
