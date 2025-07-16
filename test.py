#coding=utf-8
import torch
import torch.nn as nn
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from torch.utils.tensorboard import SummaryWriter
from utils.auto_laod_resume import auto_load_resume
from networks.model_Copy1 import MainNet
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")



#load dataset
test_transform = transforms.Compose([
        transforms.Resize((530, 530)),
        transforms.CenterCrop(512),  # 使用中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_dataset = datasets.ImageFolder(root=r'F:\paper\AttGraph\datasets\2884MARC/test', transform=test_transform)
testloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

#  Set up num_classes according to your situation
num_classes = 2884
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

#checkpoint
pth_path = r'F:\paper\AttGraph\checkpoint\best.pth'
epoch = auto_load_resume(model, pth_path, status='test')


def eval(model, testloader, criterion, status, save_path, epoch, writer=None):

    model.eval()
    print('Evaluating with softmax + auto weight search')

    best_sum_accuracy = 0.0
    best_raw_weight = 0.0
    best_local_weight = 0.0
    best_result = None

    # Store the accuracy data for plotting
    accuracy_data = []

    weight_range = [i / 10.0 for i in range(11)]  # 0.0 ~ 1.0, step size 0.1

    with torch.no_grad():
        for raw_weight in weight_range:
            local_weight = 1.0 - raw_weight

            # Initialize counters
            raw_loss_sum = 0
            local_loss_sum = 0
            total_loss_sum = 0

            top1_sum_correct = top2_sum_correct = top3_sum_correct = top4_sum_correct = top5_sum_correct = 0

            for i, data in enumerate(tqdm(testloader, desc=f"Weight raw:{raw_weight:.1f}, local:{local_weight:.1f}")):
                if status == 'CUB':
                    images, labels, boxes, scale = data
                else:
                    images, labels = data
                images, labels = images.cuda(), labels.cuda()


               # raw_logits, local_logits = model(images, epoch, i, labels, status)
                raw_logits, local_logits = model(images, status=status)
                # debug = (epoch == 0 and i == 0)
                # raw_logits, local_logits = model(images, status=status, debug=debug,
                #                                  save_path=os.path.join(save_path,
                #                                                         f"vis/test_epoch{epoch}_batch{i}.png") if debug else None)
                # Loss calculation
                raw_loss = criterion(raw_logits, labels)
                local_loss = criterion(local_logits, labels)
                total_loss = raw_loss + local_loss

                raw_loss_sum += raw_loss.item()
                local_loss_sum += local_loss.item()
                total_loss_sum += total_loss.item()

                # Softmax and weighted fusion
                raw_prob = F.softmax(raw_logits, dim=1)
                local_prob = F.softmax(local_logits, dim=1)
                sum_prob = raw_weight * raw_prob + local_weight * local_prob

                _, sum_pred = sum_prob.topk(5, 1, True, True)
                labels_expanded = labels.view(-1, 1).expand_as(sum_pred)

                top1_sum_correct += sum_pred[:, 0].eq(labels).sum().item()
                top2_sum_correct += sum_pred[:, :2].eq(labels_expanded[:, :2]).sum().item()
                top3_sum_correct += sum_pred[:, :3].eq(labels_expanded[:, :3]).sum().item()
                top4_sum_correct += sum_pred[:, :4].eq(labels_expanded[:, :4]).sum().item()
                top5_sum_correct += sum_pred.eq(labels_expanded).sum().item()

            total = len(testloader.dataset)
            sum_accuracy = [top1_sum_correct, top2_sum_correct, top3_sum_correct, top4_sum_correct, top5_sum_correct]
            sum_accuracy = [x / total * 100 for x in sum_accuracy]

            # Store the accuracy for plotting
            accuracy_data.append((raw_weight, local_weight, sum_accuracy[0]))  # Save raw_weight, local_weight, and top1 accuracy

            # Save the best weight
            if sum_accuracy[0] > best_sum_accuracy:
                best_sum_accuracy = sum_accuracy[0]
                best_raw_weight = raw_weight
                best_local_weight = local_weight
                best_result = {
                    'raw_loss': raw_loss_sum / len(testloader),
                    'local_loss': local_loss_sum / len(testloader),
                    'total_loss': total_loss_sum / len(testloader),
                    'sum_accuracy': sum_accuracy
                }

    # Output and record the best result
    result_file = os.path.join(save_path, "best_weighted_result1.txt")
    with open(result_file, "a") as f:
        f.write(f"Epoch {epoch}:\n")
        f.write(f"Best Fusion Weights -> raw: {best_raw_weight:.2f}, local: {best_local_weight:.2f}\n")
        f.write(f"Raw Loss: {best_result['raw_loss']:.4f}, Local Loss: {best_result['local_loss']:.4f}, Total Loss: {best_result['total_loss']:.4f}\n")
        f.write(f"Sum Accuracy (Top1~Top5): {' '.join([f'{x:.2f}%' for x in best_result['sum_accuracy']])}\n\n")

    # Write to TensorBoard
    if writer:
        writer.add_scalar('Loss/raw_loss', best_result['raw_loss'], epoch)
        writer.add_scalar('Loss/local_loss', best_result['local_loss'], epoch)
        writer.add_scalar('Loss/total_loss', best_result['total_loss'], epoch)
        writer.add_scalar('Accuracy/sum_accuracy', best_result['sum_accuracy'][0], epoch)

    # Plot accuracy curve (raw_weight vs. top1 accuracy)
    raw_weights = [x[0] for x in accuracy_data]
    top1_accuracies = [x[2] for x in accuracy_data]

    plt.figure()
    plt.plot(raw_weights, top1_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Raw Weight')
    plt.ylabel('Top1 Accuracy (%)')
    plt.title(f'Accuracy Curve for Epoch {epoch}')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'1accuracy_curve_epoch_{epoch}.png'))
    plt.close()

    print(f"Best raw weight: {best_raw_weight}, Best local weight: {best_local_weight}")
    print(f"Top-1 Accuracy: {best_result['sum_accuracy'][0]:.2f}%")

    return best_raw_weight, best_local_weight, best_result['sum_accuracy'][0]


if __name__ == '__main__':
    print('Testing...')

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    status = 'MACR'  # 或根据你的模型逻辑设定
    eval(model, testloader, criterion, status, save_path, epoch)
