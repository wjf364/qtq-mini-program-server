import os
import torch
import torchvision.transforms as transforms
import numpy as np
from final_zshape_dataloader import BronzeWareDataset1, BronzeWareDataset2, load_age_table1, load_age_table2, \
    load_age_table3, \
    BronzeWareDataset3
from final_zshape_setting import species_idx
# torch.cuda.current_device()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"#指定gpu


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
data_path = '../data2'
ware_img_name,ware_age, ware_shape = load_age_table1(os.path.join(data_path, 'for_1.xlsx'))
ware_shape = [species_idx[age] for age in ware_shape]
ware_shape = np.floor(np.asarray(ware_shape))
ware_age, ware_img_name, ware_shape = zip(*sorted(zip(ware_age, ware_img_name, ware_shape)))  # 对应排序
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))  # 按照顺序排好了
traindata1=ware_data
#第一批数据读取结束

#开始第二批数据
ware_img_name,ware_age, ware_shape = load_age_table2(os.path.join(data_path, 'for_2.1.xlsx'))
ware_shape = [species_idx[age] for age in ware_shape]
ware_shape = np.floor(np.asarray(ware_shape))
ware_age, ware_img_name, ware_shape = zip(*sorted(zip(ware_age, ware_img_name, ware_shape)))  # 对应排序
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))  # 按照顺序排好了
traindata21=ware_data

ware_img_name,ware_age, ware_shape = load_age_table2(os.path.join(data_path, 'for_2.2.xlsx'))
ware_shape = [species_idx[age] for age in ware_shape]
ware_shape = np.floor(np.asarray(ware_shape))
ware_age, ware_img_name, ware_shape = zip(*sorted(zip(ware_age, ware_img_name, ware_shape)))  # 对应排序
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))  # 按照顺序排好了
traindata22=ware_data

ware_img_name,ware_age, ware_shape = load_age_table2(os.path.join(data_path, 'for_2.3.xlsx'))
ware_shape = [species_idx[age] for age in ware_shape]
ware_shape = np.floor(np.asarray(ware_shape))
ware_age, ware_img_name, ware_shape = zip(*sorted(zip(ware_age, ware_img_name, ware_shape)))  # 对应排序
ware_data = np.asarray(list(zip(ware_img_name, ware_age, ware_shape)))  # 按照顺序排好了
traindata23=ware_data
#第二批数据结束

image_folder1 = os.path.join(data_path, 'ori_images_png')
image_folder21 = os.path.join(data_path, 'ori_images_png_2/1')
image_folder22 = os.path.join(data_path, 'ori_images_png_2/2')
image_folder23 = os.path.join(data_path, 'ori_images_png_2/3')
image_folder1_bg = os.path.join(data_path, 'delete_background')
image_folder21_bg = os.path.join(data_path, 'delete_background_2/1')
image_folder22_bg = os.path.join(data_path, 'delete_background_2/2')
image_folder23_bg = os.path.join(data_path, 'delete_background_2/3')
# image_folder3 = os.path.join(data_path, 'ori_images_png_3')

num=16

trainset0 = BronzeWareDataset1(image_folder1, traindata1, transform=training_preprocessing)
trainset1 = BronzeWareDataset1(image_folder1_bg, traindata1, transform=training_preprocessing)
trainset2 = BronzeWareDataset2(image_folder21, traindata21, transform=training_preprocessing)
trainset3 = BronzeWareDataset2(image_folder21_bg, traindata21, transform=training_preprocessing)
trainset4 = BronzeWareDataset2(image_folder22, traindata22, transform=training_preprocessing)
trainset5 = BronzeWareDataset2(image_folder22_bg, traindata22, transform=training_preprocessing)
trainset6 = BronzeWareDataset2(image_folder23, traindata23, transform=training_preprocessing)
trainset7 = BronzeWareDataset2(image_folder23_bg, traindata23, transform=training_preprocessing)
# trainset8 = new3_BronzeWareDataset(image_folder3_ori, traindata3, transform=training_preprocessing)
# trainset9 = new3_BronzeWareDataset(image_folder3_remove, traindata3, transform=training_preprocessing)
trainloader = torch.utils.data.DataLoader(trainset0+trainset1+trainset2+trainset3+trainset4+trainset5+trainset6+trainset7, batch_size=num, shuffle=True, num_workers=8)