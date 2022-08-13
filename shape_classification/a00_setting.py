import numpy as np


age_idx = {'商代早期': 0,
           '商代中期': 0,
           '商代晚期': 1,
           '西周早期': 2,
           '西周早期前段': 2.1,
           '西周早期后段': 2.2,
           '西周早期後段': 2.2,
           '西周中期': 3,
           '西周中期前段': 3.1,
           '西周中期後段': 3.2,
           '西周中期晚段': 3.2,
           '西周晚期': 4,
           '春秋中期': 5,
           '春秋早期': 6,
           '春秋晚期': 7,
           '戰國中期': 8,
           '戰國早期': 9,
           '戰國晚期': 10,
           '商代晚期或西周早期':11}

species_idx={
            '矮扁球腹鼎': 0,
            '半球形腹圓鼎': 1,
            '扁足方鼎': 2,
            '扁足圓鼎': 3,
            '超半球腹或半球腹鼎': 4,
            '垂腹方鼎': 5,
            '垂腹圓鼎': 6,
            '高蹄足圓鼎': 7,
            '鬲鼎': 8,
            '罐鼎': 9,
            '淺鼓腹鼎': 10,
            '收腹圓鼎': 11,
            '束腰平底鼎': 12,
            '晚期獸首蹄足鼎': 13,
            '小口鼎': 14,
            '匜鼎': 15,
            '圓鼎': 16,
            '圓錐形足圓鼎': 17,
            '早期獸首蹄足圓鼎': 18,
            '柱足方鼎': 19,
            '无': 20
}

#如果用上述feature，下面的分类是基于年代的，不能达到按照feature均匀分配的目的，
def create_dataset(dataset, division, shuffle=False):
    dataset_size = len(dataset)
    trainset_size = int(dataset_size * division[0])
    val_size = int(dataset_size * division[1])

    data_idx = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(data_idx)

    trainset = dataset[data_idx[: trainset_size]]
    valset = dataset[data_idx[trainset_size: trainset_size + val_size]]
    testset = dataset[data_idx[trainset_size + val_size:]]

    return trainset, valset, testset


def div_dataset(ware_data,division):
    j=0
    for i in range(len(ware_data)):
        if ware_data[i,1]=='1.0':
            break
    traindata0, valdata0, testdata0 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('0',j-i)
    j=i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '2.0':
            break
    traindata1, valdata1, testdata1 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('1', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '3.0':
            break
    traindata2, valdata2, testdata2 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('2', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '4.0':
            break
    traindata3, valdata3, testdata3 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('3', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '5.0':
            break
    traindata4, valdata4, testdata4 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('4', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '6.0':
            break
    traindata5, valdata5, testdata5 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('5', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '7.0':
            break
    traindata6, valdata6, testdata6 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('6', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '8.0':
            break
    traindata7, valdata7, testdata7 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('7', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '9.0':
            break
    traindata8, valdata8, testdata8 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('8', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '10.0':
            break
    traindata9, valdata9, testdata9 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('9', j - i)
    traindata10, valdata10, testdata10 = create_dataset(ware_data[i:], division, shuffle=False)
    # print('10', j - i)
    traindata=np.concatenate((traindata0,traindata1,traindata2,traindata3,traindata4,traindata5,traindata6,traindata7,traindata8,traindata9,traindata10), axis=0)
    valdata = np.concatenate((valdata0, valdata1, valdata2, valdata3, valdata4, valdata5, valdata6,valdata7, valdata8, valdata9, valdata10), axis=0)
    testdata = np.concatenate((testdata0, testdata1, testdata2, testdata3, testdata4, testdata5, testdata6,testdata7, testdata8, testdata9, testdata10), axis=0)

    return traindata,valdata,testdata

# for key,values in  age_idx.items():#同时便利键值对
#     if values==1:
#         print(key)