import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

import shape_dataset

torch.cuda.current_device()
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# from shape_dataset import traindata,testdata,test_delete_bg,test_have_bg
import shape_convnext_net


os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定gpu


def training(trainloader, testloader=None,  testloader_remove_bg=None,testloader_have_bg=None):
    # net = models.resnet18(pretrained=True).cuda()
    net = shape_convnext_net.convnext_base(pretrained=True, in_22k=True).cuda()
    # net.load_state_dict(torch.load('convnext_pth/convnext_base.pth'))

    #下方自动加权
    # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.8,0.1,0.1,0.2,0.3,0.3,0.3,0.3,0.8,0.8,0.4])).float() ,size_average=True).cuda()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([weight_age[0],weight_age[1],weight_age[2],weight_age[3],weight_age[4],weight_age[5],weight_age[6],weight_age[7],weight_age[8],weight_age[9],weight_age[10]])).float(), reduction='mean').cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
    schedulers = [MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)]

    for epoch in range(25):
        print('epoch:', epoch)

        for scheduler in schedulers:#学习率更新
            scheduler.step()
        running_loss = 0.0
        for i,data in tqdm(enumerate(trainloader)):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs.cuda())
            loss = criterion(outputs.cuda(), labels.view(-1).long().cuda())
            # loss = criterion(outputs.cuda(), labels.view(-1).long().cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        # test
        correct = 0
        total = 0

        testing(trainloader, net, 'trainset')

        if testloader is not None:
            testing(testloader, net, 'testset')

        if testloader_remove_bg is not None:
            testing(testloader_remove_bg, net, 'testloader_remove_bg')

        if testloader_have_bg is not None:
            testing(testloader_have_bg, net, 'testloader_have_bg')


    # torch.save(net, 'model/albion_grayscale_resnet101.pth')
    torch.save(net, "/home/stu/wjf/pth/1and2_convnext448_5.pth", _use_new_zipfile_serialization=True)
    print('Finished Training')


def testing(dataloader, network, which_set):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels = data
            labels = labels.view(-1).cuda()

            outputs = network(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %s images: %0.4f' % (which_set,  correct / total))


if __name__ == '__main__':
    sum=0
    weight=[0,0,0,0,0,0,0,0,0,0,0]
    trainloader=shape_dataset.trainloader
    for i, data in tqdm(enumerate(trainloader)):
        img,label_age=data
        for i in range(len(label_age.view(-1).long())):
            a = float(label_age[i])
            a = int(a)
            weight[a]=weight[a]+1
            sum=sum+1
    weight_age=[sum/weight[0],sum/weight[1],sum/weight[2],sum/weight[3],sum/weight[4],sum/weight[5],sum/weight[6],sum/weight[7],sum/weight[8],sum/weight[9],sum/weight[10]]
    testloader=shape_dataset.testloader
    test_delete_bg=shape_dataset.test_delete_bg
    test_have_bg=shape_dataset.test_have_bg
    training(trainloader, testloader,test_delete_bg,test_have_bg)
