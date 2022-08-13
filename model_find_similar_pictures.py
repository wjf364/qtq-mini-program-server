#用来找出最相似图片的文件，在同一个器型的前提下，用2048维来找出最小距离，找出最短的距离，供bronze_UI使用
#找相识图片的第二步
import torchvision.transforms as transforms
import torch.nn
from torch.autograd import Variable

import numpy as np

age_idx = {'商代早期': 0,
           '商代晚期': 1,
           '西周早期': 2,
           '西周中期': 3,
           '西周晚期': 4,
           '春秋中期': 5,
           '春秋早期': 6,
           '春秋晚期': 7,
           '战国中期': 8,
           '战国早期': 9,
           '战国晚期': 10
           }


def find_similar_pictures(predicted,pre_age,pre_shape):#前面是维度值，中间是年代值，后面是预测器形。要求输出的是同一器形且同一年代的最相似的图片
    #以下数据相互对应，存为npy是为了减少计算量，下方依次为图片的路径、经过网络的11（2048）维数据、器型（可不添加到结果）、年代（很重要）、图片对应的名字（可不添加结果）
    image_path = np.load("static/bronze_ware_data/npy/dim_path.npy")  # 下面图片对应的文件路径
    image_xml = np.load("static/bronze_ware_data/npy/dim_xml.npy")#xml文件的位置
    dimen2048 = np.load("static/bronze_ware_data/npy/dim_2048.npy",allow_pickle=True)  # 图片转化成的2048维点坐标
    ware_age = np.load("static/bronze_ware_data/npy/dim_age.npy")
    ware_shape = np.load("static/bronze_ware_data/npy/dim_shape.npy")#其实这个才是shape
    ware_birth=np.load("static/bronze_ware_data/npy/dim_birth.npy")#出土地
    ware_name=np.load("static/bronze_ware_data/npy/dim_name.npy")#器名
    ware_where=np.load("static/bronze_ware_data/npy/dim_where.npy")#出土地
    ware_chuchu = np.load("bronze_ware_data/npy/dim_chuchu.npy")  # 出处
    
    for key, values in age_idx.items():  # 同时便利键值对
        if key == pre_age:
            pre_age = values
            break
    #####直接根据相似程度找出八个即可

    min_dis8=[]#要从小到大排序
    min_dis8_path=[]
    min_dis8_xml=[]
    min_dis8_age=[]
    min_dis8_shape=[]
    min_dis8_birth = []
    min_dis8_name = []
    min_dis8_where = []
    min_dis8_chuchu=[]

    k=0#用来定位path
    for i in dimen2048:
        if pre_shape == ware_shape[k] and int(pre_age)==int(float(ware_age[k])):#同一器形才输出
            dis=0
            for j in range(len(i)):
                dis=pow(predicted[0][j]-i[j],2)+dis
            if len(min_dis8)<8:
                min_dis8.append(dis)
                min_dis8_path.append(image_path[k])
                min_dis8_xml.append(image_xml[k])
                min_dis8_age.append(ware_age[k])
                min_dis8_shape.append(ware_shape[k])
                min_dis8_birth.append(ware_birth[k])
                min_dis8_name.append(ware_name[k])
                min_dis8_where.append(ware_where[k])
                min_dis8_chuchu.append(ware_chuchu[k])
                #排序
                m=len(min_dis8)-1
                while m > 0:
                    if min_dis8[m] < min_dis8[m - 1]:
                        temp = min_dis8[m]
                        min_dis8[m] = min_dis8[m - 1]
                        min_dis8[m - 1] = temp
                        temp = min_dis8_path[m]
                        min_dis8_path[m] = min_dis8_path[m - 1]
                        min_dis8_path[m - 1] = temp
                        temp = min_dis8_xml[m]
                        min_dis8_xml[m] = min_dis8_xml[m - 1]
                        min_dis8_xml[m - 1] = temp
                        temp = min_dis8_age[m]
                        min_dis8_age[m] = min_dis8_age[m - 1]
                        min_dis8_age[m - 1] = temp
                        temp = min_dis8_shape[m]
                        min_dis8_shape[m] = min_dis8_shape[m - 1]
                        min_dis8_shape[m - 1] = temp
                        temp = min_dis8_birth[m]
                        min_dis8_birth[m] = min_dis8_birth[m - 1]
                        min_dis8_birth[m - 1] = temp
                        temp = min_dis8_name[m]
                        min_dis8_name[m] = min_dis8_name[m - 1]
                        min_dis8_name[m - 1] = temp
                        temp = min_dis8_where[m]
                        min_dis8_where[m] = min_dis8_where[m - 1]
                        min_dis8_where[m - 1] = temp
                        temp = min_dis8_chuchu[m]
                        min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                        min_dis8_chuchu[m - 1] = temp
                    else:
                        break
                    m = m - 1
            else:
                if dis<min_dis8[-1]:#小于最后一个点，把最后一个点替换掉，再进行排序
                    min_dis8[-1]=dis
                    min_dis8_path[-1]=image_path[k]
                    min_dis8_xml[-1] = image_xml[k]
                    min_dis8_age[-1]=ware_age[k]
                    min_dis8_shape[-1]=ware_shape[k]
                    min_dis8_birth[-1]=ware_birth[k]
                    min_dis8_name[-1]=ware_name[k]
                    min_dis8_where[-1]=ware_where[k]
                    min_dis8_chuchu[-1] = ware_chuchu[k]
                    #排序
                    m=7
                    while m>0:
                        if min_dis8[m]<min_dis8[m-1]:
                            temp=min_dis8[m]
                            min_dis8[m]=min_dis8[m-1]
                            min_dis8[m-1]=temp
                            temp = min_dis8_path[m]
                            min_dis8_path[m] = min_dis8_path[m - 1]
                            min_dis8_path[m - 1] = temp
                            temp = min_dis8_xml[m]
                            min_dis8_xml[m] = min_dis8_xml[m - 1]
                            min_dis8_xml[m - 1] = temp
                            temp = min_dis8_age[m]
                            min_dis8_age[m] = min_dis8_age[m - 1]
                            min_dis8_age[m - 1] = temp
                            temp = min_dis8_shape[m]
                            min_dis8_shape[m] = min_dis8_shape[m - 1]
                            min_dis8_shape[m - 1] = temp
                            temp = min_dis8_birth[m]
                            min_dis8_birth[m] = min_dis8_birth[m - 1]
                            min_dis8_birth[m - 1] = temp
                            temp = min_dis8_name[m]
                            min_dis8_name[m] = min_dis8_name[m - 1]
                            min_dis8_name[m - 1] = temp
                            temp = min_dis8_where[m]
                            min_dis8_where[m] = min_dis8_where[m - 1]
                            min_dis8_where[m - 1] = temp
                            temp = min_dis8_chuchu[m]
                            min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                            min_dis8_chuchu[m - 1] = temp
                        else:
                            break
                        m=m-1
        k=k+1

    # print(len(min_dis8))
    if len(min_dis8)<8:#证明找出的图片的个数小于8个，从其它途径添加图片
        #先在同器型里面添加图片，限制条件：相同器型而且最相似（剔除已经有的）
        k = 0  # 用来定位path
        for i in dimen2048:
            if pre_shape == ware_shape[k] and image_path[k] not in min_dis8_path:  # 同一器形才输出
                dis = 0
                for j in range(len(i)):
                    dis = pow(predicted[0][j] - i[j], 2) + dis
                if len(min_dis8) < 8:
                    min_dis8.append(dis)
                    min_dis8_path.append(image_path[k])
                    min_dis8_xml.append(image_xml[k])
                    min_dis8_age.append(ware_age[k])
                    min_dis8_shape.append(ware_shape[k])
                    min_dis8_birth.append(ware_birth[k])
                    min_dis8_name.append(ware_name[k])
                    min_dis8_where.append(ware_where[k])
                    min_dis8_chuchu.append(ware_chuchu[k])
                    # 排序
                    m = len(min_dis8) - 1
                    while m > 0:
                        if min_dis8[m] < min_dis8[m - 1]:
                            temp = min_dis8[m]
                            min_dis8[m] = min_dis8[m - 1]
                            min_dis8[m - 1] = temp
                            temp = min_dis8_path[m]
                            min_dis8_path[m] = min_dis8_path[m - 1]
                            min_dis8_path[m - 1] = temp
                            temp = min_dis8_xml[m]
                            min_dis8_xml[m] = min_dis8_xml[m - 1]
                            min_dis8_xml[m - 1] = temp
                            temp = min_dis8_age[m]
                            min_dis8_age[m] = min_dis8_age[m - 1]
                            min_dis8_age[m - 1] = temp
                            temp = min_dis8_shape[m]
                            min_dis8_shape[m] = min_dis8_shape[m - 1]
                            min_dis8_shape[m - 1] = temp
                            temp = min_dis8_birth[m]
                            min_dis8_birth[m] = min_dis8_birth[m - 1]
                            min_dis8_birth[m - 1] = temp
                            temp = min_dis8_name[m]
                            min_dis8_name[m] = min_dis8_name[m - 1]
                            min_dis8_name[m - 1] = temp
                            temp = min_dis8_where[m]
                            min_dis8_where[m] = min_dis8_where[m - 1]
                            min_dis8_where[m - 1] = temp
                            temp = min_dis8_chuchu[m]
                            min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                            min_dis8_chuchu[m - 1] = temp
                        else:
                            break
                        m = m - 1
                else:
                    if dis < min_dis8[-1]:  # 小于最后一个点，把最后一个点替换掉，再进行排序
                        min_dis8[-1] = dis
                        min_dis8_path[-1] = image_path[k]
                        min_dis8_xml[-1] = image_xml[k]
                        min_dis8_age[-1] = ware_age[k]
                        min_dis8_shape[-1] = ware_shape[k]
                        min_dis8_birth[-1] = ware_birth[k]
                        min_dis8_name[-1] = ware_name[k]
                        min_dis8_where[-1] = ware_where[k]
                        min_dis8_chuchu[-1] = ware_chuchu[k]
                        # 排序
                        m = 7
                        while m > 0:
                            if min_dis8[m] < min_dis8[m - 1]:
                                temp = min_dis8[m]
                                min_dis8[m] = min_dis8[m - 1]
                                min_dis8[m - 1] = temp
                                temp = min_dis8_path[m]
                                min_dis8_path[m] = min_dis8_path[m - 1]
                                min_dis8_path[m - 1] = temp
                                temp = min_dis8_xml[m]
                                min_dis8_xml[m] = min_dis8_xml[m - 1]
                                min_dis8_xml[m - 1] = temp
                                temp = min_dis8_age[m]
                                min_dis8_age[m] = min_dis8_age[m - 1]
                                min_dis8_age[m - 1] = temp
                                temp = min_dis8_shape[m]
                                min_dis8_shape[m] = min_dis8_shape[m - 1]
                                min_dis8_shape[m - 1] = temp
                                temp = min_dis8_birth[m]
                                min_dis8_birth[m] = min_dis8_birth[m - 1]
                                min_dis8_birth[m - 1] = temp
                                temp = min_dis8_name[m]
                                min_dis8_name[m] = min_dis8_name[m - 1]
                                min_dis8_name[m - 1] = temp
                                temp = min_dis8_where[m]
                                min_dis8_where[m] = min_dis8_where[m - 1]
                                min_dis8_where[m - 1] = temp
                                temp = min_dis8_chuchu[m]
                                min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                                min_dis8_chuchu[m - 1] = temp
                            else:
                                break
                            m = m - 1
            k = k + 1

    if len(min_dis8)<8:#证明找出的图片的个数小于8个，从其它途径添加图片
        #再在同年代里面添加图片，限制条件：相同年代而且最相似（剔除已经有的）
        k = 0  # 用来定位path
        for i in dimen2048:
            if int(pre_age)==int(float(ware_age[k])) and image_path[k] not in min_dis8_path:  # 同一器形才输出
                dis = 0
                for j in range(len(i)):
                    dis = pow(predicted[0][j] - i[j], 2) + dis
                if len(min_dis8) < 8:
                    min_dis8.append(dis)
                    min_dis8_path.append(image_path[k])
                    min_dis8_xml.append(image_xml[k])
                    min_dis8_age.append(ware_age[k])
                    min_dis8_shape.append(ware_shape[k])
                    min_dis8_birth.append(ware_birth[k])
                    min_dis8_name.append(ware_name[k])
                    min_dis8_where.append(ware_where[k])
                    min_dis8_chuchu.append(ware_chuchu[k])
                    # 排序
                    m = len(min_dis8) - 1
                    while m > 0:
                        if min_dis8[m] < min_dis8[m - 1]:
                            temp = min_dis8[m]
                            min_dis8[m] = min_dis8[m - 1]
                            min_dis8[m - 1] = temp
                            temp = min_dis8_path[m]
                            min_dis8_path[m] = min_dis8_path[m - 1]
                            min_dis8_path[m - 1] = temp
                            temp = min_dis8_xml[m]
                            min_dis8_xml[m] = min_dis8_xml[m - 1]
                            min_dis8_xml[m - 1] = temp
                            temp = min_dis8_age[m]
                            min_dis8_age[m] = min_dis8_age[m - 1]
                            min_dis8_age[m - 1] = temp
                            temp = min_dis8_shape[m]
                            min_dis8_shape[m] = min_dis8_shape[m - 1]
                            min_dis8_shape[m - 1] = temp
                            temp = min_dis8_birth[m]
                            min_dis8_birth[m] = min_dis8_birth[m - 1]
                            min_dis8_birth[m - 1] = temp
                            temp = min_dis8_name[m]
                            min_dis8_name[m] = min_dis8_name[m - 1]
                            min_dis8_name[m - 1] = temp
                            temp = min_dis8_where[m]
                            min_dis8_where[m] = min_dis8_where[m - 1]
                            min_dis8_where[m - 1] = temp
                            temp = min_dis8_chuchu[m]
                            min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                            min_dis8_chuchu[m - 1] = temp
                        else:
                            break
                        m = m - 1
                else:
                    if dis < min_dis8[-1]:  # 小于最后一个点，把最后一个点替换掉，再进行排序
                        min_dis8[-1] = dis
                        min_dis8_path[-1] = image_path[k]
                        min_dis8_xml[-1] = image_xml[k]
                        min_dis8_age[-1] = ware_age[k]
                        min_dis8_shape[-1] = ware_shape[k]
                        min_dis8_birth[-1] = ware_birth[k]
                        min_dis8_name[-1] = ware_name[k]
                        min_dis8_where[-1] = ware_where[k]
                        min_dis8_chuchu[-1] = ware_chuchu[k]
                        # 排序
                        m = 7
                        while m > 0:
                            if min_dis8[m] < min_dis8[m - 1]:
                                temp = min_dis8[m]
                                min_dis8[m] = min_dis8[m - 1]
                                min_dis8[m - 1] = temp
                                temp = min_dis8_path[m]
                                min_dis8_path[m] = min_dis8_path[m - 1]
                                min_dis8_path[m - 1] = temp
                                temp = min_dis8_xml[m]
                                min_dis8_xml[m] = min_dis8_xml[m - 1]
                                min_dis8_xml[m - 1] = temp
                                temp = min_dis8_age[m]
                                min_dis8_age[m] = min_dis8_age[m - 1]
                                min_dis8_age[m - 1] = temp
                                temp = min_dis8_shape[m]
                                min_dis8_shape[m] = min_dis8_shape[m - 1]
                                min_dis8_shape[m - 1] = temp
                                temp = min_dis8_birth[m]
                                min_dis8_birth[m] = min_dis8_birth[m - 1]
                                min_dis8_birth[m - 1] = temp
                                temp = min_dis8_name[m]
                                min_dis8_name[m] = min_dis8_name[m - 1]
                                min_dis8_name[m - 1] = temp
                                temp = min_dis8_where[m]
                                min_dis8_where[m] = min_dis8_where[m - 1]
                                min_dis8_where[m - 1] = temp
                                temp = min_dis8_chuchu[m]
                                min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                                min_dis8_chuchu[m - 1] = temp
                            else:
                                break
                            m = m - 1
            k = k + 1

    if len(min_dis8)<8:#证明找出的图片的个数小于8个，从其它途径添加图片
        #最后只加上最相似的即可（剔除已经有的）
        k = 0  # 用来定位path
        for i in dimen2048:
            if image_path[k] not in min_dis8_path:  # 同一器形才输出
                dis = 0
                for j in range(len(i)):
                    dis = pow(predicted[0][j] - i[j], 2) + dis
                if len(min_dis8) < 8:
                    min_dis8.append(dis)
                    min_dis8_path.append(image_path[k])
                    min_dis8_xml.append(image_xml[k])
                    min_dis8_age.append(ware_age[k])
                    min_dis8_shape.append(ware_shape[k])
                    min_dis8_birth.append(ware_birth[k])
                    min_dis8_name.append(ware_name[k])
                    min_dis8_where.append(ware_where[k])
                    min_dis8_chuchu.append(ware_chuchu[k])
                    # 排序
                    m = len(min_dis8) - 1
                    while m > 0:
                        if min_dis8[m] < min_dis8[m - 1]:
                            temp = min_dis8[m]
                            min_dis8[m] = min_dis8[m - 1]
                            min_dis8[m - 1] = temp
                            temp = min_dis8_path[m]
                            min_dis8_path[m] = min_dis8_path[m - 1]
                            min_dis8_path[m - 1] = temp
                            temp = min_dis8_xml[m]
                            min_dis8_xml[m] = min_dis8_xml[m - 1]
                            min_dis8_xml[m - 1] = temp
                            temp = min_dis8_age[m]
                            min_dis8_age[m] = min_dis8_age[m - 1]
                            min_dis8_age[m - 1] = temp
                            temp = min_dis8_shape[m]
                            min_dis8_shape[m] = min_dis8_shape[m - 1]
                            min_dis8_shape[m - 1] = temp
                            temp = min_dis8_birth[m]
                            min_dis8_birth[m] = min_dis8_birth[m - 1]
                            min_dis8_birth[m - 1] = temp
                            temp = min_dis8_name[m]
                            min_dis8_name[m] = min_dis8_name[m - 1]
                            min_dis8_name[m - 1] = temp
                            temp = min_dis8_where[m]
                            min_dis8_where[m] = min_dis8_where[m - 1]
                            min_dis8_where[m - 1] = temp
                            temp = min_dis8_chuchu[m]
                            min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                            min_dis8_chuchu[m - 1] = temp
                        else:
                            break
                        m = m - 1
                else:
                    if dis < min_dis8[-1]:  # 小于最后一个点，把最后一个点替换掉，再进行排序
                        min_dis8[-1] = dis
                        min_dis8_path[-1] = image_path[k]
                        min_dis8_xml[-1] = image_xml[k]
                        min_dis8_age[-1] = ware_age[k]
                        min_dis8_shape[-1] = ware_shape[k]
                        min_dis8_birth[-1] = ware_birth[k]
                        min_dis8_name[-1] = ware_name[k]
                        min_dis8_where[-1] = ware_where[k]
                        min_dis8_chuchu[-1] = ware_chuchu[k]
                        # 排序
                        m = 7
                        while m > 0:
                            if min_dis8[m] < min_dis8[m - 1]:
                                temp = min_dis8[m]
                                min_dis8[m] = min_dis8[m - 1]
                                min_dis8[m - 1] = temp
                                temp = min_dis8_path[m]
                                min_dis8_path[m] = min_dis8_path[m - 1]
                                min_dis8_path[m - 1] = temp
                                temp = min_dis8_xml[m]
                                min_dis8_xml[m] = min_dis8_xml[m - 1]
                                min_dis8_xml[m - 1] = temp
                                temp = min_dis8_age[m]
                                min_dis8_age[m] = min_dis8_age[m - 1]
                                min_dis8_age[m - 1] = temp
                                temp = min_dis8_shape[m]
                                min_dis8_shape[m] = min_dis8_shape[m - 1]
                                min_dis8_shape[m - 1] = temp
                                temp = min_dis8_birth[m]
                                min_dis8_birth[m] = min_dis8_birth[m - 1]
                                min_dis8_birth[m - 1] = temp
                                temp = min_dis8_name[m]
                                min_dis8_name[m] = min_dis8_name[m - 1]
                                min_dis8_name[m - 1] = temp
                                temp = min_dis8_where[m]
                                min_dis8_where[m] = min_dis8_where[m - 1]
                                min_dis8_where[m - 1] = temp
                                temp = min_dis8_chuchu[m]
                                min_dis8_chuchu[m] = min_dis8_chuchu[m - 1]
                                min_dis8_chuchu[m - 1] = temp
                            else:
                                break
                            m = m - 1
            k = k + 1
    # print(len(min_dis8))

    return min_dis8,min_dis8_path,min_dis8_xml,min_dis8_age,min_dis8_shape,min_dis8_name,min_dis8_birth,min_dis8_where,min_dis8_chuchu