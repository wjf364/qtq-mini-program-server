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
           '战国中期': 8,
           '战国早期': 9,
           '战国晚期': 10,
           '戰國中期': 8,
           '戰國早期': 9,
           '戰國晚期': 10
           }

shape_idx={}
#如果用上述feature，下面的分类是基于年代的，不能达到按照feature均匀分配的目的，0.2:0.8
def create_dataset(dataset, division, shuffle=False):
    dataset_size = len(dataset)
    trainset_size = int(dataset_size * division[0])
    val_size = int(dataset_size * division[1])#0.8

    data_idx = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(data_idx)

    # # 0-20
    # trainset = dataset[data_idx[: trainset_size]]
    # valset = dataset[data_idx[trainset_size:]]
    # return trainset, valset

    # # # 20-40
    # trainset1 = dataset[data_idx[: trainset_size]]
    # trainset2 = dataset[data_idx[trainset_size: trainset_size+trainset_size]]
    # valset = dataset[data_idx[trainset_size+trainset_size:]]
    #
    # return trainset2,np.concatenate((trainset1,valset),axis=0)

    # # 40-60
    # trainset1 = dataset[data_idx[: trainset_size+trainset_size]]
    # trainset2 = dataset[data_idx[trainset_size+trainset_size: trainset_size+trainset_size+trainset_size]]
    # valset = dataset[data_idx[trainset_size+trainset_size+trainset_size:]]
    #
    # return trainset2,np.concatenate((trainset1,valset),axis=0)

    # # # 60-80
    # trainset1 = dataset[data_idx[: trainset_size+trainset_size+trainset_size]]
    # trainset2 = dataset[data_idx[trainset_size+trainset_size+trainset_size: trainset_size+trainset_size+trainset_size+trainset_size]]
    # valset = dataset[data_idx[trainset_size+trainset_size+trainset_size+trainset_size:]]
    #
    # return trainset2,np.concatenate((trainset1,valset),axis=0)

    # # 80-100
    trainset1 = dataset[data_idx[: trainset_size+trainset_size+trainset_size+trainset_size]]
    trainset2 = dataset[data_idx[trainset_size+trainset_size+trainset_size+trainset_size: ]]

    return trainset2,trainset1


def div_dataset(ware_data,division):
    j=0
    for i in range(len(ware_data)):
        if ware_data[i,1]=='1.0':
            break
    traindata0,  testdata0 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('0',j-i)
    j=i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '2.0':
            break
    traindata1,  testdata1 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('1', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '3.0':
            break
    traindata2, testdata2 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('2', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '4.0':
            break
    traindata3,  testdata3 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('3', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '5.0':
            break
    traindata4, testdata4 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('4', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '6.0':
            break
    traindata5,  testdata5 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('5', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '7.0':
            break
    traindata6,  testdata6 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('6', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '8.0':
            break
    traindata7, testdata7 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('7', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '9.0':
            break
    traindata8,  testdata8 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('8', j - i)
    j = i
    for i in range(len(ware_data)):
        if ware_data[i, 1] == '10.0':
            break
    traindata9,  testdata9 = create_dataset(ware_data[j:i], division, shuffle=False)
    # print('9', j - i)
    traindata10,  testdata10 = create_dataset(ware_data[i:], division, shuffle=False)
    # print('10', j - i)
    traindata=np.concatenate((traindata0,traindata1,traindata2,traindata3,traindata4,traindata5,traindata6,traindata7,traindata8,traindata9,traindata10), axis=0)
    testdata = np.concatenate((testdata0, testdata1, testdata2, testdata3, testdata4, testdata5, testdata6,testdata7, testdata8, testdata9, testdata10), axis=0)

    return traindata,testdata

# for key,values in  age_idx.items():#同时便利键值对
#     if values==1:
#         print(key)