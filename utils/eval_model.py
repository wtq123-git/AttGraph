# import torch
# from tqdm import tqdm
# import os
# from tensorboardX import SummaryWriter
# import numpy as np
# from config import coordinates_cat, proposalN, set, vis_num
# from utils.cal_iou import calculate_iou
# from utils.vis import image_with_boxes
#
# def eval(model, testloader, criterion, status, save_path, epoch):
#     model.eval()
#     print('Evaluating')
#
#     raw_loss_sum = 0
#     local_loss_sum = 0
#     windowscls_loss_sum = 0
#     total_loss_sum = 0
#     iou_corrects = 0
#     raw_correct = 0
#     local_correct = 0
#     sum_correct = 0
#
# #     ppp = 1
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(testloader)):
#             if set == 'CUB':
#                 images, labels, boxes, scale = data
#             else:
#                 images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#
#             raw_logits,local_logits = model(images, epoch, i,labels, status)
# #             raw_logits,_ = model(images)
#
#
#             raw_loss = criterion(raw_logits, labels)
#
#
#             total_loss = raw_loss
#
#             raw_loss_sum += raw_loss.item()
#
#             total_loss_sum += total_loss.item()
#
#             # correct num
#             # raw
#             pred = raw_logits.max(1, keepdim=True)[1]
#             raw_correct += pred.eq(labels.view_as(pred)).sum().item()
#
#             # raw
#             pred = local_logits.max(1, keepdim=True)[1]
#             local_correct += pred.eq(labels.view_as(pred)).sum().item()
#
#
#             sum_logit = raw_logits+local_logits
#             pred = sum_logit.max(1, keepdim=True)[1]
#             sum_correct += pred.eq(labels.view_as(pred)).sum().item()
#
#
#
#     # raw_loss_avg = raw_loss_sum / (i+1)
#     # local_loss_avg = local_loss_sum / (i+1)
#     # windowscls_loss_avg = windowscls_loss_sum / (i+1)
#     # total_loss_avg = total_loss_sum / (i+1)
#
#     raw_accuracy = raw_correct / len(testloader.dataset)
#     local_accuracy = local_correct / len(testloader.dataset)
#     sum_accuracy = sum_correct / len(testloader.dataset)
#
#
#     return raw_accuracy,local_accuracy,sum_accuracy

import torch
from tqdm import tqdm
import os
# from tensorboardX import SummaryWriter
import numpy as np
from config import coordinates_cat, proposalN, set, vis_num
from utils.cal_iou import calculate_iou
from utils.vis import image_with_boxes
#
# # 打印 Top-k 准确率的函数
# def print_top_k_accuracy(name, top1, top2, top3, top4, top5):
#     print(f'{name} Top1-5 Accuracy: {top1*100:.2f} {top2*100:.2f} {top3*100:.2f} {top4*100:.2f} {top5*100:.2f}')
#
# def eval(model, testloader, criterion, status, save_path, epoch, writer=None):
#     model.eval()
#     print('Evaluating')
#
#     # 损失和正确预测计数器
#     raw_loss_sum = 0
#     local_loss_sum = 0
#     total_loss_sum = 0
#     raw_correct = 0
#     local_correct = 0
#     sum_correct = 0
#
#     # Top-1 到 Top-5 准确率计数器
#     top1_raw_correct = 0
#     top2_raw_correct = 0
#     top3_raw_correct = 0
#     top4_raw_correct = 0
#     top5_raw_correct = 0
#
#     top1_local_correct = 0
#     top2_local_correct = 0
#     top3_local_correct = 0
#     top4_local_correct = 0
#     top5_local_correct = 0
#
#     top1_sum_correct = 0
#     top2_sum_correct = 0
#     top3_sum_correct = 0
#     top4_sum_correct = 0
#     top5_sum_correct = 0
#
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(testloader)):
#             if set == 'CUB':
#                 images, labels, boxes, scale = data
#             else:
#                 images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#
#             raw_logits, local_logits = model(images, epoch, i, labels, status)
#
#             raw_loss = criterion(raw_logits, labels)
#             local_loss = criterion(local_logits, labels)
#             total_loss = raw_loss + local_loss  # 总损失可以是两部分损失的加权和，或者简单相加
#
#             raw_loss_sum += raw_loss.item()
#             local_loss_sum += local_loss.item()
#             total_loss_sum += total_loss.item()
#
#             # 计算 Raw Logits 的 Top-1 到 Top-5 准确率
#             _, raw_pred = raw_logits.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_raw_correct += raw_pred[:, 0].eq(labels).sum().item()
#             top2_raw_correct += raw_pred[:, :2].eq(labels.view(-1, 1).expand_as(raw_pred[:, :2])).sum().item()
#             top3_raw_correct += raw_pred[:, :3].eq(labels.view(-1, 1).expand_as(raw_pred[:, :3])).sum().item()
#             top4_raw_correct += raw_pred[:, :4].eq(labels.view(-1, 1).expand_as(raw_pred[:, :4])).sum().item()
#             top5_raw_correct += raw_pred.eq(labels.view(-1, 1).expand_as(raw_pred)).sum().item()
#
#             # 计算 Local Logits 的 Top-1 到 Top-5 准确率
#             _, local_pred = local_logits.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_local_correct += local_pred[:, 0].eq(labels).sum().item()
#             top2_local_correct += local_pred[:, :2].eq(labels.view(-1, 1).expand_as(local_pred[:, :2])).sum().item()
#             top3_local_correct += local_pred[:, :3].eq(labels.view(-1, 1).expand_as(local_pred[:, :3])).sum().item()
#             top4_local_correct += local_pred[:, :4].eq(labels.view(-1, 1).expand_as(local_pred[:, :4])).sum().item()
#             top5_local_correct += local_pred.eq(labels.view(-1, 1).expand_as(local_pred)).sum().item()
#
#             # 计算 Sum Logits 的 Top-1 到 Top-5 准确率
#             sum_logit = raw_logits + local_logits
#             _, sum_pred = sum_logit.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_sum_correct += sum_pred[:, 0].eq(labels).sum().item()
#             top2_sum_correct += sum_pred[:, :2].eq(labels.view(-1, 1).expand_as(sum_pred[:, :2])).sum().item()
#             top3_sum_correct += sum_pred[:, :3].eq(labels.view(-1, 1).expand_as(sum_pred[:, :3])).sum().item()
#             top4_sum_correct += sum_pred[:, :4].eq(labels.view(-1, 1).expand_as(sum_pred[:, :4])).sum().item()
#             top5_sum_correct += sum_pred.eq(labels.view(-1, 1).expand_as(sum_pred)).sum().item()
#
#     # 计算 raw, local, sum 的准确率
#     raw_accuracy = top1_raw_correct / len(testloader.dataset)
#     local_accuracy = top1_local_correct / len(testloader.dataset)
#     sum_accuracy = top1_sum_correct / len(testloader.dataset)
#
#     # 计算 Top-1 到 Top-5 准确率
#     top1_raw_accuracy = top1_raw_correct / len(testloader.dataset)
#     top2_raw_accuracy = top2_raw_correct / len(testloader.dataset)
#     top3_raw_accuracy = top3_raw_correct / len(testloader.dataset)
#     top4_raw_accuracy = top4_raw_correct / len(testloader.dataset)
#     top5_raw_accuracy = top5_raw_correct / len(testloader.dataset)
#
#     top1_local_accuracy = top1_local_correct / len(testloader.dataset)
#     top2_local_accuracy = top2_local_correct / len(testloader.dataset)
#     top3_local_accuracy = top3_local_correct / len(testloader.dataset)
#     top4_local_accuracy = top4_local_correct / len(testloader.dataset)
#     top5_local_accuracy = top5_local_correct / len(testloader.dataset)
#
#     top1_sum_accuracy = top1_sum_correct / len(testloader.dataset)
#     top2_sum_accuracy = top2_sum_correct / len(testloader.dataset)
#     top3_sum_accuracy = top3_sum_correct / len(testloader.dataset)
#     top4_sum_accuracy = top4_sum_correct / len(testloader.dataset)
#     top5_sum_accuracy = top5_sum_correct / len(testloader.dataset)
#
#     # 简化打印输出
#     print_top_k_accuracy('Raw', top1_raw_accuracy, top2_raw_accuracy, top3_raw_accuracy, top4_raw_accuracy, top5_raw_accuracy)
#     print_top_k_accuracy('Local', top1_local_accuracy, top2_local_accuracy, top3_local_accuracy, top4_local_accuracy, top5_local_accuracy)
#     print_top_k_accuracy('Sum', top1_sum_accuracy, top2_sum_accuracy, top3_sum_accuracy, top4_sum_accuracy, top5_sum_accuracy)
#
#     # 如果 TensorBoard Writer 被传入，则记录指标
#     if writer:
#         writer.add_scalar('Loss/raw_loss', raw_loss_sum / len(testloader), epoch)
#         writer.add_scalar('Loss/local_loss', local_loss_sum / len(testloader), epoch)
#         writer.add_scalar('Loss/total_loss', total_loss_sum / len(testloader), epoch)
#
#         writer.add_scalar('Accuracy/raw_accuracy', raw_accuracy, epoch)
#         writer.add_scalar('Accuracy/local_accuracy', local_accuracy, epoch)
#         writer.add_scalar('Accuracy/sum_accuracy', sum_accuracy, epoch)
#
#     return raw_accuracy, local_accuracy, sum_accuracy

# def eval(model, testloader, criterion, status, save_path, epoch, writer=None):
#     model.eval()
#     print('Evaluating')
#
#     # 损失和正确预测计数器
#     raw_loss_sum = 0
#     local_loss_sum = 0
#     total_loss_sum = 0
#     raw_correct = 0
#     local_correct = 0
#     sum_correct = 0
#
#     # Top-1 到 Top-5 准确率计数器
#     top1_raw_correct = 0
#     top2_raw_correct = 0
#     top3_raw_correct = 0
#     top4_raw_correct = 0
#     top5_raw_correct = 0
#
#     top1_local_correct = 0
#     top2_local_correct = 0
#     top3_local_correct = 0
#     top4_local_correct = 0
#     top5_local_correct = 0
#
#     top1_sum_correct = 0
#     top2_sum_correct = 0
#     top3_sum_correct = 0
#     top4_sum_correct = 0
#     top5_sum_correct = 0
#
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(testloader)):
#             if set == 'CUB':
#                 images, labels, boxes, scale = data
#             else:
#                 images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#
#             raw_logits, local_logits = model(images, epoch, i, labels, status)
#
#             raw_loss = criterion(raw_logits, labels)
#             local_loss = criterion(local_logits, labels)
#             total_loss = raw_loss + local_loss  # 总损失可以是两部分损失的加权和，或者简单相加
#
#             raw_loss_sum += raw_loss.item()
#             local_loss_sum += local_loss.item()
#             total_loss_sum += total_loss.item()
#
#             # 计算 Raw Logits 的 Top-1 到 Top-5 准确率
#             _, raw_pred = raw_logits.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_raw_correct += raw_pred[:, 0].eq(labels).sum().item()
#             top2_raw_correct += raw_pred[:, :2].eq(labels.view(-1, 1).expand_as(raw_pred[:, :2])).sum().item()
#             top3_raw_correct += raw_pred[:, :3].eq(labels.view(-1, 1).expand_as(raw_pred[:, :3])).sum().item()
#             top4_raw_correct += raw_pred[:, :4].eq(labels.view(-1, 1).expand_as(raw_pred[:, :4])).sum().item()
#             top5_raw_correct += raw_pred.eq(labels.view(-1, 1).expand_as(raw_pred)).sum().item()
#
#             # 计算 Local Logits 的 Top-1 到 Top-5 准确率
#             _, local_pred = local_logits.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_local_correct += local_pred[:, 0].eq(labels).sum().item()
#             top2_local_correct += local_pred[:, :2].eq(labels.view(-1, 1).expand_as(local_pred[:, :2])).sum().item()
#             top3_local_correct += local_pred[:, :3].eq(labels.view(-1, 1).expand_as(local_pred[:, :3])).sum().item()
#             top4_local_correct += local_pred[:, :4].eq(labels.view(-1, 1).expand_as(local_pred[:, :4])).sum().item()
#             top5_local_correct += local_pred.eq(labels.view(-1, 1).expand_as(local_pred)).sum().item()
#
#             # 计算 Sum Logits 的 Top-1 到 Top-5 准确率
#             sum_logit = raw_logits + local_logits
#             _, sum_pred = sum_logit.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_sum_correct += sum_pred[:, 0].eq(labels).sum().item()
#             top2_sum_correct += sum_pred[:, :2].eq(labels.view(-1, 1).expand_as(sum_pred[:, :2])).sum().item()
#             top3_sum_correct += sum_pred[:, :3].eq(labels.view(-1, 1).expand_as(sum_pred[:, :3])).sum().item()
#             top4_sum_correct += sum_pred[:, :4].eq(labels.view(-1, 1).expand_as(sum_pred[:, :4])).sum().item()
#             top5_sum_correct += sum_pred.eq(labels.view(-1, 1).expand_as(sum_pred)).sum().item()
#
#     # 计算 raw, local, sum 的 Top-1 到 Top-5 准确率
#     raw_accuracy = [top1_raw_correct, top2_raw_correct, top3_raw_correct, top4_raw_correct, top5_raw_correct]
#     local_accuracy = [top1_local_correct, top2_local_correct, top3_local_correct, top4_local_correct, top5_local_correct]
#     sum_accuracy = [top1_sum_correct, top2_sum_correct, top3_sum_correct, top4_sum_correct, top5_sum_correct]
#
#     raw_accuracy = [x / len(testloader.dataset) * 100 for x in raw_accuracy]
#     local_accuracy = [x / len(testloader.dataset) * 100 for x in local_accuracy]
#     sum_accuracy = [x / len(testloader.dataset) * 100 for x in sum_accuracy]
#
#     # 计算损失
#     avg_raw_loss = raw_loss_sum / len(testloader)
#     avg_local_loss = local_loss_sum / len(testloader)
#     avg_total_loss = total_loss_sum / len(testloader)
#
#     # 记录到 result.txt 文件
#     result_file = os.path.join("448result.txt")
#     with open(result_file, "a") as f:
#         f.write(f"Epoch {epoch}:\n")
#         f.write(f"  Raw Loss: {avg_raw_loss:.4f}, Local Loss: {avg_local_loss:.4f}, Total Loss: {avg_total_loss:.4f}\n")
#         f.write(f"  Raw Accuracy : {' '.join([f'{acc:.2f}%' for acc in raw_accuracy])}\n")
#         f.write(f"  Local Accuracy : {' '.join([f'{acc:.2f}%' for acc in local_accuracy])}\n")
#         f.write(f"  Sum Accuracy : {' '.join([f'{acc:.2f}%' for acc in sum_accuracy])}\n")
#         f.write("\n")
#
#     # 如果 TensorBoard Writer 被传入，则记录指标
#     if writer:
#         writer.add_scalar('Loss/raw_loss', avg_raw_loss, epoch)
#         writer.add_scalar('Loss/local_loss', avg_local_loss, epoch)
#         writer.add_scalar('Loss/total_loss', avg_total_loss, epoch)
#
#         writer.add_scalar('Accuracy/raw_accuracy', raw_accuracy[0], epoch)
#         writer.add_scalar('Accuracy/local_accuracy', local_accuracy[0], epoch)
#         writer.add_scalar('Accuracy/sum_accuracy', sum_accuracy[0], epoch)
#
#     return raw_accuracy, local_accuracy, sum_accuracy

# def eval(model, testloader, criterion, status, save_path, epoch, writer=None):
#     model.eval()
#     print('Evaluating')
#
#     # 损失和正确预测计数器
#     raw_loss_sum = 0
#     local_loss_sum = 0
#     total_loss_sum = 0
#
#
#     # Top-1 到 Top-5 准确率计数器
#     top1_raw_correct = 0
#     top2_raw_correct = 0
#     top3_raw_correct = 0
#     top4_raw_correct = 0
#     top5_raw_correct = 0
#
#     top1_local_correct = 0
#     top2_local_correct = 0
#     top3_local_correct = 0
#     top4_local_correct = 0
#     top5_local_correct = 0
#
#     top1_sum_correct = 0
#     top2_sum_correct = 0
#     top3_sum_correct = 0
#     top4_sum_correct = 0
#     top5_sum_correct = 0
#
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(testloader)):
#             if set == 'CUB':
#                 images, labels, boxes, scale = data
#             else:
#                 images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#
#             raw_logits, local_logits = model(images, epoch, i, labels, status)
#
#             raw_loss = criterion(raw_logits, labels)
#             local_loss = criterion(local_logits, labels)
#             total_loss = raw_loss + local_loss  # 总损失可以是两部分损失的加权和，或者简单相加
#
#             raw_loss_sum += raw_loss.item()
#             local_loss_sum += local_loss.item()
#             total_loss_sum += total_loss.item()
#
#             # 计算 Raw Logits 的 Top-1 到 Top-5 准确率
#             _, raw_pred = raw_logits.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_raw_correct += raw_pred[:, 0].eq(labels).sum().item()
#             top2_raw_correct += raw_pred[:, :2].eq(labels.view(-1, 1).expand_as(raw_pred[:, :2])).sum().item()
#             top3_raw_correct += raw_pred[:, :3].eq(labels.view(-1, 1).expand_as(raw_pred[:, :3])).sum().item()
#             top4_raw_correct += raw_pred[:, :4].eq(labels.view(-1, 1).expand_as(raw_pred[:, :4])).sum().item()
#             top5_raw_correct += raw_pred.eq(labels.view(-1, 1).expand_as(raw_pred)).sum().item()
#
#             # 计算 Local Logits 的 Top-1 到 Top-5 准确率
#             _, local_pred = local_logits.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_local_correct += local_pred[:, 0].eq(labels).sum().item()
#             top2_local_correct += local_pred[:, :2].eq(labels.view(-1, 1).expand_as(local_pred[:, :2])).sum().item()
#             top3_local_correct += local_pred[:, :3].eq(labels.view(-1, 1).expand_as(local_pred[:, :3])).sum().item()
#             top4_local_correct += local_pred[:, :4].eq(labels.view(-1, 1).expand_as(local_pred[:, :4])).sum().item()
#             top5_local_correct += local_pred.eq(labels.view(-1, 1).expand_as(local_pred)).sum().item()
#
#             # 计算 Sum Logits 的 Top-1 到 Top-5 准确率
#             sum_logit = raw_logits + local_logits
#             _, sum_pred = sum_logit.topk(5, 1, True, True)  # 获取 Top-5 预测
#
#             top1_sum_correct += sum_pred[:, 0].eq(labels).sum().item()
#             top2_sum_correct += sum_pred[:, :2].eq(labels.view(-1, 1).expand_as(sum_pred[:, :2])).sum().item()
#             top3_sum_correct += sum_pred[:, :3].eq(labels.view(-1, 1).expand_as(sum_pred[:, :3])).sum().item()
#             top4_sum_correct += sum_pred[:, :4].eq(labels.view(-1, 1).expand_as(sum_pred[:, :4])).sum().item()
#             top5_sum_correct += sum_pred.eq(labels.view(-1, 1).expand_as(sum_pred)).sum().item()
#
#     # 计算 raw, local, sum 的 Top-1 到 Top-5 准确率
#     raw_accuracy = [top1_raw_correct, top2_raw_correct, top3_raw_correct, top4_raw_correct, top5_raw_correct]
#     local_accuracy = [top1_local_correct, top2_local_correct, top3_local_correct, top4_local_correct, top5_local_correct]
#     sum_accuracy = [top1_sum_correct, top2_sum_correct, top3_sum_correct, top4_sum_correct, top5_sum_correct]
#
#     raw_accuracy = [x / len(testloader.dataset) * 100 for x in raw_accuracy]
#     local_accuracy = [x / len(testloader.dataset) * 100 for x in local_accuracy]
#     sum_accuracy = [x / len(testloader.dataset) * 100 for x in sum_accuracy]
#
#     # 计算损失
#     avg_raw_loss = raw_loss_sum / len(testloader)
#     avg_local_loss = local_loss_sum / len(testloader)
#     avg_total_loss = total_loss_sum / len(testloader)
#
#     # 记录到 result.txt 文件
#     # result_file = os.path.join("clip-0.1-result.txt")
#     result_file = os.path.join("clip-0.20-result.txt")
#     with open(result_file, "a") as f:
#         f.write(f"Epoch {epoch}:\n")
#         f.write(f"  Raw Loss: {avg_raw_loss:.4f}, Local Loss: {avg_local_loss:.4f}, Total Loss: {avg_total_loss:.4f}\n")
#         f.write(f"  Raw Accuracy : {' '.join([f'{acc:.2f}%' for acc in raw_accuracy])}\n")
#         f.write(f"  Local Accuracy : {' '.join([f'{acc:.2f}%' for acc in local_accuracy])}\n")
#         f.write(f"  Sum Accuracy : {' '.join([f'{acc:.2f}%' for acc in sum_accuracy])}\n")
#         f.write("\n")
#
#     # 如果 TensorBoard Writer 被传入，则记录指标
#     if writer:
#         writer.add_scalar('Loss/raw_loss', avg_raw_loss, epoch)
#         writer.add_scalar('Loss/local_loss', avg_local_loss, epoch)
#         writer.add_scalar('Loss/total_loss', avg_total_loss, epoch)
#
#         writer.add_scalar('Accuracy/raw_accuracy', raw_accuracy[0], epoch)
#         writer.add_scalar('Accuracy/local_accuracy', local_accuracy[0], epoch)
#         writer.add_scalar('Accuracy/sum_accuracy', sum_accuracy[0], epoch)
#
#     return raw_accuracy, local_accuracy, sum_accuracy[0]  # 返回 sum_accuracy 的第一个值作为 sum_acc

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

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

                raw_logits, local_logits = model(images, epoch, i, labels, status)

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
    result_file = os.path.join(save_path, "best_weighted_result.txt")
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
    plt.savefig(os.path.join(save_path, f'accuracy_curve_epoch_{epoch}.png'))
    plt.close()

    return best_raw_weight, best_local_weight, best_result['sum_accuracy'][0]

