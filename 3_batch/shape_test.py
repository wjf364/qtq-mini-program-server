#测试本年代预测分布的，例如真实值为年代1，%多少预测为年代1，%多少预测为年代2，%多少预测为年代3，取最高的三个（）
import os
import torch
import t_convnext_net
torch.cuda.current_device()
import torchvision
import torchvision.transforms as transforms
import t_dataset_for_test
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定gpu

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


validation_preprocessing = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])


def testing(dataloader, network, which_set):

    with torch.no_grad():
        total=0
        correct=0
        num=0
        img_path=[]
        for data in tqdm(dataloader):
            inputs,path, labels = data
            labels = labels.view(-1).cuda()

            outputs = network(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            num2,predicted2=torch.topk(outputs.data, k=2, dim=1)
            total += labels.size(0)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:#原始判断条件,这个条件满足自动忽略下一条件
                    correct=correct+1
                else:#不相等有一次挽救的机会
                    a = predicted2[i][0]
                    b=predicted2[i][1]
                    a1=float(a)
                    b1=float(b)
                    a=int(a1)
                    b=int(b1)

                    if a==1 and b==2:#找到模糊图片了
                        img_path.append(path[i])
                    elif a==2 and b==1:
                        img_path.append(path[i])

    print(img_path,len(img_path))
    print('#############################总体的测试准确率为(商代晚期和西周早期出现一个商代晚期或西周早期)：')
    print('Accuracy of the network on the %s images: %d %%' % (which_set, 100 * correct / total))

#这个用来找出分类正确但是很接近的
def testing_close(dataloader, network, which_set):

    with torch.no_grad():
        total=0
        correct=0
        num=0
        img_path=[]#100~90%
        img_path1 = []#90~85%
        img_path2 = []  # 85~80%
        img_path3 = []  # 80~60%
        img_path4 = []  # 60~40%
        img_path5 = []  # 40~20%
        img_path6 = []  # 20~10%
        img_path7 = []  # 10~5%
        img_path8 = []  # 5~0%
        for data in tqdm(dataloader):
            inputs,path, labels = data
            labels = labels.view(-1).cuda()

            outputs = network(inputs.cuda())
            _, predicted = torch.max(outputs.data, 1)
            num2,predicted2=torch.topk(outputs.data, k=2, dim=1)
            total += labels.size(0)
            print(num2)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:#预测正确，且非常接近
                    correct=correct+1
                    a = predicted2[i][0]#大的
                    b=predicted2[i][1]#小的
                    c=num2[i][0]#大的
                    d=num2[i][1]#小的
                    # print(c,d,d/c)
                    a1=float(a)
                    b1=float(b)
                    a=int(a1)
                    b=int(b1)

                    if a==1 and b==2:#找到可以比较相邻的
                        if d/c>=0.9:#大于0.9，我可以认为他们是接近的
                            img_path.append(path[i])
                        if d / c >= 0.85 and d/c<0.9:  # 大于0.9，我可以认为他们是接近的
                            img_path1.append(path[i])
                        if d / c >= 0.8 and d/c<0.85:  # 大于0.9，我可以认为他们是接近的
                            img_path2.append(path[i])
                        if d / c >= 0.6 and d / c < 0.8:  # 大于0.9，我可以认为他们是接近的
                            img_path3.append(path[i])
                        if d / c >= 0.4 and d / c < 0.6:  # 大于0.9，我可以认为他们是接近的
                            img_path4.append(path[i])
                        if d / c >= 0.2 and d / c < 0.4:  # 大于0.9，我可以认为他们是接近的
                            img_path5.append(path[i])
                        if d / c >= 0.1 and d / c < 0.2:  # 大于0.9，我可以认为他们是接近的
                            img_path6.append(path[i])
                        if d / c >= 0.05 and d / c < 0.1:  # 大于0.9，我可以认为他们是接近的
                            img_path7.append(path[i])
                        if d / c < 0.05:  # 大于0.9，我可以认为他们是接近的
                            img_path8.append(path[i])
                    elif a==2 and b==1:
                        # print(d/c)
                        if d / c >= 0.9:  # 大于0.9，我可以认为他们是接近的
                            img_path.append(path[i])
                        if d / c >= 0.85 and d / c < 0.9:  # 大于0.9，我可以认为他们是接近的
                            img_path1.append(path[i])
                        if d / c >= 0.8 and d / c < 0.85:  # 大于0.9，我可以认为他们是接近的
                            img_path2.append(path[i])
                        if d / c >= 0.6 and d / c < 0.8:  # 大于0.9，我可以认为他们是接近的
                            img_path3.append(path[i])
                        if d / c >= 0.4 and d / c < 0.6:  # 大于0.9，我可以认为他们是接近的
                            img_path4.append(path[i])
                        if d / c >= 0.2 and d / c < 0.4:  # 大于0.9，我可以认为他们是接近的
                            img_path5.append(path[i])
                        if d / c >= 0.1 and d / c < 0.2:  # 大于0.9，我可以认为他们是接近的
                            img_path6.append(path[i])
                        if d / c >= 0.05 and d / c < 0.1:  # 大于0.9，我可以认为他们是接近的
                            img_path7.append(path[i])
                        if d / c < 0.05:  # 大于0.9，我可以认为他们是接近的
                            img_path8.append(path[i])
    print('100~90%',img_path)
    print('90~85%', img_path1)
    print('85~80%', img_path2)
    print('80~60%', img_path3)
    print('60~40%', img_path4)
    print('40~20%', img_path5)
    print('20~10%', img_path6)
    print('10~5%', img_path7)
    print('5~0%', img_path8)
    print('#############################总体的测试准确率为(商代晚期和西周早期出现一个商代晚期或西周早期)：')
    print('Accuracy of the network on the %s images: %d %%' % (which_set, 100 * correct / total))

if __name__ == '__main__':
    net = t_convnext_net.convnext_base(pretrained=True, in_22k=True).cuda()
    trained_weight = torch.load("/home/stu/wjf/pth/1and2_convnext448_5.pth")
    net.load_state_dict(trained_weight.state_dict())  # 竟然这么容易就可以加载部分预训练模型，将strict改了就可以

    testloader=t_dataset_for_test.test_for_find
    testing_close(testloader, net, 'test_for_find')