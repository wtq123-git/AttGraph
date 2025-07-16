# coding=utf-8
import torch
import random
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, set, channels
from utils.train_model import train
from utils.auto_laod_resume import auto_load_resume
from networks.model_Copy1 import MainNet
from networks.convnext import convnext_tiny

import os
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def count_params(model):
    """
    计算并打印模型的总参数量
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total model parameters: {total_params:,}')
    return total_params

def main():
    # 加载数据
    #12.26 修改数据构建，使用ImageFolder 114-128
    train_transform = transforms.Compose([
        transforms.Resize((530, 530)),
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((530, 530)),
        transforms.CenterCrop(512),  # 使用中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=r'F:\paper\AttGraph\datasets\2884MARC/train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root=r'F:\paper\AttGraph\datasets\2884MARC/test', transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
    # trainloader, testloader = read_dataset(input_size, batch_size, root, set)

    # Debug: Check original labels in the dataset
    # for i in range(5):
    #     print(f"Image Path: {train_dataset.imgs[i][0]}, Label: {train_dataset.imgs[i][1]}")
    #     print(f"Image Path: {test_dataset.imgs[i][0]}, Label: {test_dataset.imgs[i][1]}")

    # Print batch-wise label and file paths to ensure correct loading
    # for i, (x, y) in enumerate(trainloader):  # Assuming trainloader for training data
    #     print(f"Batch {i}, File Paths: {x}, Labels: {y}")

    # 定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
    # model = convnext_tiny(pretrained=True, num_classes=2884)

    # 打印模型的参数量
    count_params(model)

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()

    # 加载checkpoint
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr

    # 定义优化器
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    model = model.cuda()  # 部署在GPU

    # 定义学习率调度器
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir="512AttGraph/logs")

    # 开始训练
    train(model=model,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval,
          writer=writer)  # 将 writer 传递给 train 函数

    # 关闭 writer
    writer.close()


def setup_seed(seed):  # 确保实验的可重复性和结果的稳定性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    #     torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    setup_seed(12345)
    main()
