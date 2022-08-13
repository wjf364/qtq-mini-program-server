import os
import torch
import torchvision.transforms as transforms
import numpy as np
from t_dataloader import BronzeWareDataset, new_BronzeWareDataset, load_age_table1, load_age_table2, load_age_table3, \
    new3_BronzeWareDataset
from t_setting import age_idx1, create_dataset, div_dataset
torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定gpu


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

training_preprocessing = transforms.Compose([
    transforms.RandomResizedCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

validation_preprocessing = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])

#读取第一批数据
data_path = 'data2'
ware_img_name,ware_age, ware_shape = load_age_table1(os.path.join(data_path, 'for_1.xlsx'))
#处理不规范数据
for idx in range(len(ware_age)):
    if ware_age[idx][-1] == '\u3000':
        ware_age[idx] = ware_age[idx][:-1]
ware_age = [age_idx1[age] for age in ware_age]
ware_age = np.floor(np.asarray(ware_age))
ware_age, ware_img_name, ware_shape = zip(*sorted(zip(ware_age, ware_img_name, ware_shape)))  # 对应排序
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))  # 按照顺序排好了
# traindata, testdata = div_dataset(ware_data, [0.8, 0.2])
testdata,  traindata = div_dataset(ware_data, [0.2, 0.8])
#第一批数据读取结束

#开始处理第二批，共三个部分
#21
ware_img_name21,ware_age21, ware_shape21 = load_age_table2(os.path.join(data_path, 'for_2.1.xlsx'))
for idx in range(len(ware_age21)):
    if ware_age21[idx][-1] == '\u3000':
        ware_age21[idx] = ware_age21[idx][:-1]
ware_age21 = [age_idx1[age] for age in ware_age21]
ware_age21 = np.floor(np.asarray(ware_age21))
ware_age21, ware_img_name21,ware_shape21 = zip(*sorted(zip(ware_age21, ware_img_name21,ware_shape21)))  # 对应排序
ware_data21 = np.asarray(list(zip(ware_img_name21, ware_age21,ware_shape21)))  # 按照顺序排好了
# traindata21,  testdata21 = div_dataset(ware_data21, [0.8, 0.2])
testdata21,  traindata21 = div_dataset(ware_data21, [0.2, 0.8])
#22
ware_img_name22,ware_age22, ware_shape22 = load_age_table2(os.path.join(data_path, 'for_2.2.xlsx'))
for idx in range(len(ware_age22)):
    if ware_age22[idx][-1] == '\u3000':
        ware_age22[idx] = ware_age22[idx][:-1]
ware_age22 = [age_idx1[age] for age in ware_age22]
ware_age22 = np.floor(np.asarray(ware_age22))
ware_age22, ware_img_name22,ware_shape22 = zip(*sorted(zip(ware_age22, ware_img_name22,ware_shape22)))  # 对应排序
ware_data22 = np.asarray(list(zip(ware_img_name22, ware_age22,ware_shape22)))  # 按照顺序排好了
# traindata22,  testdata22 = div_dataset(ware_data22, [0.8, 0.2])
testdata22,  traindata22 = div_dataset(ware_data22, [0.2, 0.8])
#23
ware_img_name23,ware_age23, ware_shape23 = load_age_table2(os.path.join(data_path, 'for_2.3.xlsx'))
for idx in range(len(ware_age23)):
    if ware_age23[idx][-1] == '\u3000':
        ware_age23[idx] = ware_age23[idx][:-1]
ware_age23 = [age_idx1[age] for age in ware_age23]
ware_age23 = np.floor(np.asarray(ware_age23))
ware_age23, ware_img_name23,ware_shape23 = zip(*sorted(zip(ware_age23, ware_img_name23,ware_shape23)))  # 对应排序
ware_data23 = np.asarray(list(zip(ware_img_name23, ware_age23,ware_shape23)))  # 按照顺序排好了
# traindata23,  testdata23 = div_dataset(ware_data23, [0.8, 0.2])
testdata23,  traindata23 = div_dataset(ware_data23, [0.2, 0.8])
#第二批数据处理也结束

#开始处理第三批数据
ware_img_name3,ware_age3, ware_shape3 = load_age_table3(os.path.join(data_path, 'for_3_delete_150.xlsx'))
for idx in range(len(ware_age3)):
    if ware_age3[idx][-1] == '\u3000':
        ware_age3[idx] = ware_age3[idx][:-1]
    if ware_age3[idx][-1] == '\xa0':
        ware_age3[idx] = ware_age3[idx][:-1]
ware_age3 = [age_idx1[age] for age in ware_age3]
#剔除label=11的，也就是提出商代晚期或西周早期
pos = 0
no_pos = []
no_pos_alte=[]#反向处理，要记得从后面开始减才行
for i in ware_age3:
    pos = pos + 1
    if i == 11:
        no_pos.append(pos - 1)
j=len(no_pos)-1
while j>=0:
    no_pos_alte.append(no_pos[j])
    j=j-1
for i in no_pos_alte:
    ware_age3 = np.delete(ware_age3, i)
    ware_shape3 = np.delete(ware_shape3, i)
    ware_img_name3 = np.delete(ware_img_name3, i)
for i in ware_age3:#验证删除结果
    if i == 11:
        print('删除错误')
#剔除结束
ware_age3 = np.floor(np.asarray(ware_age3))
ware_age3, ware_img_name3,ware_shape3 = zip(*sorted(zip(ware_age3, ware_img_name3,ware_shape3)))  # 对应排序
ware_data3 = np.asarray(list(zip(ware_img_name3, ware_age3,ware_shape3)))  # 按照顺序排好了
# traindata3,  testdata3 = div_dataset(ware_data3, [0.8, 0.2])
testdata3,  traindata3 = div_dataset(ware_data3, [0.2, 0.8])
#处理第三批数据结束
# 数据处理真正结束

#图片路径，后面数字代表第几批数据
#原本的图片
image_folder1_ori = 'data2/ori_images_png'
image_folder21_ori='data2/ori_images_png_2/1'
image_folder22_ori='data2/ori_images_png_2/2'
image_folder23_ori='data2/ori_images_png_2/3'
image_folder3_ori = 'data2/ori_images_png_3'
#去背景的图片
image_folder1_remove = 'data2/delete_background'
image_folder21_remove ='data2/delete_background_2/1'
image_folder22_remove ='data2/delete_background_2/2'
image_folder23_remove ='data2/delete_background_2/3'
image_folder3_remove = 'data2/delete_background_3'
#灰度图片
image_folder1_gray = 'data2/gray'
image_folder21_gray='data2/gray_2/1'
image_folder22_gray='data2/gray_2/2'
image_folder23_gray='data2/gray_2/3'
image_folder3_gray = 'data2/gray_3'
#线图
image_folder1_line = 'data2/image_line'
image_folder21_line='data2/image_line_2/1'
image_folder22_line='data2/image_line_2/2'
image_folder23_line='data2/image_line_2/3'
image_folder3_line = 'data2/image_line_3'
# load dataset
num=12

trainset0 = BronzeWareDataset(image_folder1_ori, traindata, transform=training_preprocessing)
trainset1 = BronzeWareDataset(image_folder1_remove, traindata, transform=training_preprocessing)
trainset2 = new_BronzeWareDataset(image_folder21_ori, traindata21, transform=training_preprocessing)
trainset3 = new_BronzeWareDataset(image_folder21_remove, traindata21, transform=training_preprocessing)
trainset4 = new_BronzeWareDataset(image_folder22_ori, traindata22, transform=training_preprocessing)
trainset5 = new_BronzeWareDataset(image_folder22_remove, traindata22, transform=training_preprocessing)
trainset6 = new_BronzeWareDataset(image_folder23_ori, traindata23, transform=training_preprocessing)
trainset7 = new_BronzeWareDataset(image_folder23_remove, traindata23, transform=training_preprocessing)
trainset8 = new3_BronzeWareDataset(image_folder3_ori, traindata3, transform=training_preprocessing)
trainset9 = new3_BronzeWareDataset(image_folder3_remove, traindata3, transform=training_preprocessing)
# trainloader = torch.utils.data.DataLoader(trainset + trainset1, batch_size=num,shuffle=True, num_workers=8)
trainloader = torch.utils.data.DataLoader(trainset0+trainset1+trainset2+trainset3+trainset4+trainset5+trainset6+trainset7+trainset8+trainset9, batch_size=num, shuffle=True, num_workers=8)


testset0 = BronzeWareDataset(image_folder1_ori, testdata, transform=validation_preprocessing)
testset1 = BronzeWareDataset(image_folder1_remove, testdata, transform=validation_preprocessing)
testset2 = new_BronzeWareDataset(image_folder21_ori, testdata21, transform=validation_preprocessing)
testset3 = new_BronzeWareDataset(image_folder21_remove, testdata21, transform=validation_preprocessing)
testset4 = new_BronzeWareDataset(image_folder22_ori, testdata22, transform=validation_preprocessing)
testset5 = new_BronzeWareDataset(image_folder22_remove, testdata22, transform=validation_preprocessing)
testset6 = new_BronzeWareDataset(image_folder23_ori, testdata23, transform=validation_preprocessing)
testset7 = new_BronzeWareDataset(image_folder23_remove, testdata23, transform=validation_preprocessing)
testset8 = new3_BronzeWareDataset(image_folder3_ori, testdata3, transform=validation_preprocessing)
testset9 = new3_BronzeWareDataset(image_folder3_remove, testdata3, transform=validation_preprocessing)
# testloader = torch.utils.data.DataLoader(testset+testset1, batch_size=num, shuffle=False, num_workers=4)
testloader = torch.utils.data.DataLoader(testset0+testset1+testset2+testset3+testset4+testset5+testset6+testset7+testset8+testset9, batch_size=num, shuffle=False, num_workers=4)
test_delete_bg = torch.utils.data.DataLoader(testset1, batch_size=num, shuffle=False, num_workers=4)
test_have_bg = torch.utils.data.DataLoader(testset0, batch_size=num, shuffle=False, num_workers=4)
test_for_find = torch.utils.data.DataLoader(testset0+testset2+testset4+testset6, batch_size=num, shuffle=False, num_workers=4)