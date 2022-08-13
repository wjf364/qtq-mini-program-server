import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

import final_zshape_dataset

torch.cuda.current_device()
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import final_zshape_convnext_net


os.environ["CUDA_VISIBLE_DEVICES"] = "3"#指定gpu


def training(trainloader, testloader=None,  testloader_remove_bg=None,testloader_have_bg=None):
    # net = models.resnet18(pretrained=True).cuda()
    net = final_zshape_convnext_net.convnext_base(pretrained=True, in_22k=True).cuda()
    # net.load_state_dict(torch.load('convnext_pth/convnext_base.pth'))

    #下方自动加权
    # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.8,0.1,0.1,0.2,0.3,0.3,0.3,0.3,0.8,0.8,0.4])).float() ,size_average=True).cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([weight_age[0],weight_age[1],weight_age[2],weight_age[3],weight_age[4],weight_age[5],weight_age[6],weight_age[7],weight_age[8],weight_age[9],weight_age[10],weight_age[11],weight_age[12],weight_age[13],weight_age[14],weight_age[15],weight_age[16],weight_age[17],weight_age[18],weight_age[19],weight_age[20],weight_age[21],weight_age[22],weight_age[23],weight_age[24],weight_age[25],weight_age[26],weight_age[27],weight_age[28],weight_age[29]])).float(), reduction='mean').cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    schedulers = [MultiStepLR(optimizer, milestones=[17, 22], gamma=0.1)]

    for epoch in range(25):
        print('epoch:', epoch)

        for scheduler in schedulers:#学习率更新
            scheduler.step()
        running_loss = 0.0
        for i,data in tqdm(enumerate(trainloader)):
            inputs, shape = data
            optimizer.zero_grad()

            outputs = net(inputs.cuda())
            loss = criterion(outputs.cuda(), shape.view(-1).long().cuda())
            # loss = criterion(outputs.cuda(), labels.view(-1).long().cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))
        testing(trainloader, net, 'trainset')


    torch.save(net, "pth/final_shape.pth", _use_new_zipfile_serialization=True)
    print('Finished Training')


def testing(dataloader, network, which_set):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, shape = data
            shape = shape.view(-1).cuda()

            outputs = network(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += shape.size(0)
            correct += (predicted == shape).sum().item()

    print('Accuracy of the network on the %s images: %0.4f' % (which_set,  correct / total))


if __name__ == '__main__':
    sum=0
    weight=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    trainloader=final_zshape_dataset.trainloader
    for i, data in tqdm(enumerate(trainloader)):
        img,shape=data
        for i in range(len(shape.view(-1).long())):
            a = float(shape[i])
            a = int(a)
            weight[a]=weight[a]+1
            sum=sum+1
    weight_age=[sum/weight[0],sum/weight[1],sum/weight[2],sum/weight[3],sum/weight[4],sum/weight[5],sum/weight[6],sum/weight[7],sum/weight[8],sum/weight[9],sum/weight[10],sum/weight[11],sum/weight[12],sum/weight[13],sum/weight[14],sum/weight[15],sum/weight[16],sum/weight[17],sum/weight[18],sum/weight[19],sum/weight[20],sum/weight[21],sum/weight[22],sum/weight[23],sum/weight[24],sum/weight[25],sum/weight[26],sum/weight[27],sum/weight[28],sum/weight[29]]
    training(trainloader)
