import os
import torch
import transforms
from network_files import SparseRCNN, SparseRCNNPredictor
from backbone import resnet50_fpn_backbone
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
import zz_dataset


def create_model(num_classes):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = SparseRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    # weights_dict = torch.load("fast_rcnn/pth/retinanet_resnet50_fpn_coco.pth", map_location='cpu')
    weights_dict = torch.load("pth/resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = SparseRCNNPredictor(in_features, num_classes)

    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    #下面三个不能删除
    aspect_ratio_group_factor = 3
    batch_size = 2
    amp = False  # 是否使用混合精度训练，需要GPU支持

    train_dataset = zz_dataset.trainset1+zz_dataset.trainset21+zz_dataset.trainset22+zz_dataset.trainset23+zz_dataset.trainset3
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=zz_dataset.trainset1.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=zz_dataset.trainset1.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=13)#背景+12个年代，裁剪框和图片本身的label相互对应
    # print(model)

    model.to(device)

    # define optimizer,判断哪些部分是需要训练的
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    scaler = torch.cuda.amp.GradScaler() if amp else None

    # learning rate scheduler，逐步降低学习率
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    train_loss = []
    learning_rate = []

    num_epochs=80

    for epoch in range(num_epochs):
        print(epoch)
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        # update the learning rate
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()

    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    if amp:
        save_files["scaler"] = scaler.state_dict()
    torch.save(save_files, "pth/object_wenshi.pth".format(epoch),_use_new_zipfile_serialization=True)


if __name__ == "__main__":
    main()
