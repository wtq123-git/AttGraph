# import os
# import glob
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
# from config import max_checkpoint_num, proposalN, eval_trainset, set
# from utils.eval_model import eval
# # from networks.resnet import *
# import torch.nn.functional as F
#
# class LabelSmoothLoss(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super(LabelSmoothLoss, self).__init__()
#         self.smoothing = smoothing
#
#     def forward(self, input, target):
#         log_prob = F.log_softmax(input, dim=-1)
#         weight = input.new_ones(input.size()) * \
#                  self.smoothing / (input.size(-1) - 1.)
#         weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
#         loss = (-weight * log_prob).sum(dim=-1).mean()
#         return loss
#
# class CenterLoss(nn.Module):
#     """Center loss.
#
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
#
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """
#     def __init__(self, num_classes=200, feat_dim=768, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu
#
#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
#
#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())
#
#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))
#
#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
#
#         return loss
#
# def con_loss_new(features, labels):
#     eps = 1e-6
#
#     B, _ = features.shape
#     features = F.normalize(features)
#     cos_matrix = features.mm(features.t())
#
#     pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
#     neg_label_matrix = 1 - pos_label_matrix
#
#     neg_label_matrix_new = 1 - pos_label_matrix
#
#     pos_cos_matrix = 1 - cos_matrix
#     neg_cos_matrix =1 + cos_matrix
#
#
#     margin = 0.3
#
#
#     sim = (1 + cos_matrix)/2.0
#     scores = 1 - sim
#
#
#     positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores-scores)
#     mask = torch.eye(features.size(0)).cuda()
#     positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)
#
#     #print(positive_scores)
#     #print(torch.sum(positive_scores, dim=1, keepdim=True))
#     #print(torch.sum(pos_label_matrix, dim=1, keepdim=True)-1)
#
#     positive_scores = torch.sum(positive_scores, dim=1, keepdim=True)/((torch.sum(pos_label_matrix, dim=1, keepdim=True)-1)+eps)
#     positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)
#
#     #print(positive_scores)
#
#     relative_dis1 = margin + positive_scores -scores
#     neg_label_matrix_new[relative_dis1 < 0] = 0
#     neg_label_matrix = neg_label_matrix*neg_label_matrix_new
#
#     #print(neg_label_matrix)
#
#
#
#     loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
#     loss /= B*B
#
#     #print(loss)
#
#     #print('---------------------')
#
#     return loss
#
#
# def train(model,
#           trainloader,
#           testloader,
#           criterion,
#           optimizer,
#           scheduler,
#           save_path,
#           start_epoch,
#           end_epoch,
#           save_interval):
#     last_acc = 0
#     for epoch in range(start_epoch + 1, end_epoch + 1):
#
#
#         model.train()
#
#         print('Training %d epoch' % epoch)
#
#         lr = next(iter(optimizer.param_groups))['lr']
#
#         for i, data in enumerate(tqdm(trainloader)):
#             if set == 'CUB':
#                 images, labels, _, _ = data
#             else:
#                 images, labels = data
#             images, labels = images.cuda(), labels.cuda()
#
#             optimizer.zero_grad()
#             # 有梯度的
#             raw_logits,local_logits = model(images, epoch, i,'train')
# #             raw_logits,_ = model(images)
#
#
#
#             raw_loss = criterion(raw_logits, labels)
#             local_loss = criterion(local_logits, labels)
#
# #             contra_loss = con_loss_new(fim,labels)
#
#
#             total_loss = raw_loss + local_loss
#
#             #total_loss.requires_grad_(True)
#             total_loss.backward()
#
#             optimizer.step()
#
#         scheduler.step()
#
#
#
#         # eval testset
#         raw_accuracy,local_accuracy,sum_accuracy = eval(model, testloader, criterion, 'test', save_path, epoch)
#
#         print(
#             'Test set: raw accuracy: {:.2f}%'.format(100. * raw_accuracy))
#         print(
#             'Test set: local accuracy: {:.2f}%'.format(100. * local_accuracy))
#         print(
#             'Test set: sum accuracy: {:.2f}%'.format(100. * sum_accuracy))
#
#         # save checkpoint
#         if (epoch % save_interval == 0) or (epoch == end_epoch):
#             print('Saving checkpoint')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'learning_rate': lr,
#             }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
#
#
#         # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
#         # and delete the redundant ones
#         checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
#         if len(checkpoint_list) == max_checkpoint_num + 1:
#             idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
#             min_idx = min(idx_list)
#             os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))
#


# import glob
# import os
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter  # 确保使用正确的导入方式
# import torch.nn.functional as F
# from config import max_checkpoint_num, proposalN, eval_trainset, set
# from utils.eval_model import eval
#
#
# class LabelSmoothLoss(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super(LabelSmoothLoss, self).__init__()
#         self.smoothing = smoothing
#
#     def forward(self, input, target):
#         log_prob = F.log_softmax(input, dim=-1)
#         weight = input.new_ones(input.size()) * \
#                  self.smoothing / (input.size(-1) - 1.)
#         weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
#         loss = (-weight * log_prob).sum(dim=-1).mean()
#         return loss
#
#
# class CenterLoss(nn.Module):
#     """Center loss.
#
#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
#
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#         use_gpu (bool): whether to use GPU.
#     """
#
#     def __init__(self, num_classes=200, feat_dim=768, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu
#
#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
#
#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())
#
#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))
#
#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
#
#         return loss
#
#
# def con_loss_new(features, labels):
#     eps = 1e-6
#
#     B, _ = features.shape
#     features = F.normalize(features)
#     cos_matrix = features.mm(features.t())
#
#     pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
#     neg_label_matrix = 1 - pos_label_matrix
#
#     neg_label_matrix_new = 1 - pos_label_matrix
#
#     pos_cos_matrix = 1 - cos_matrix
#     neg_cos_matrix = 1 + cos_matrix
#
#     margin = 0.3
#
#     sim = (1 + cos_matrix) / 2.0
#     scores = 1 - sim
#
#     positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores - scores)
#     mask = torch.eye(features.size(0)).cuda()
#     positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)
#
#     positive_scores = torch.sum(positive_scores, dim=1, keepdim=True) / (
#                 (torch.sum(pos_label_matrix, dim=1, keepdim=True) - 1) + eps)
#     positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)
#
#     relative_dis1 = margin + positive_scores - scores
#     neg_label_matrix_new[relative_dis1 < 0] = 0
#     neg_label_matrix = neg_label_matrix * neg_label_matrix_new
#
#     loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
#     loss /= B * B
#
#     return loss
#
#
# def train(model,
#           trainloader,
#           testloader,
#           criterion,
#           optimizer,
#           scheduler,
#           save_path,
#           start_epoch,
#           end_epoch,
#           save_interval,
#           writer=None):  # 接受 writer 作为参数
#     last_acc = 0
#     for epoch in range(start_epoch + 1, end_epoch + 1):
#         model.train()
#
#         print('Training %d epoch' % epoch)
#         lr = next(iter(optimizer.param_groups))['lr']
#
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for i, data in enumerate(tqdm(trainloader)):
#             if set == 'CUB':
#                 images, labels, _, _ = data
#             else:
#                 images, labels = data
#             images, labels = images.cuda(), labels.cuda()
#
#             optimizer.zero_grad()
#             raw_logits, local_logits = model(images, epoch, i, 'train')
#
#             raw_loss = criterion(raw_logits, labels)
#             local_loss = criterion(local_logits, labels)
#
#             total_loss = raw_loss + local_loss
#             total_loss.backward()
#
#             optimizer.step()
#
#             running_loss += total_loss.item()
#
#             # 计算准确度
#             _, predicted = torch.max(raw_logits, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             # 每10个批次记录一次
#             if writer is not None and i % 10 == 0:
#                 writer.add_scalar('training_loss', running_loss / (i + 1), epoch * len(trainloader) + i)
#                 writer.add_scalar('training_accuracy', 100. * correct / total, epoch * len(trainloader) + i)
#
#                 # 记录输入图像（如果需要）
#                 if i < 5:  # 每个epoch记录前5个批次的图像
#                     writer.add_images('train_images', images, epoch * 100 + i)
#
#         # 每个epoch结束后，记录损失和准确率
#         if writer is not None:
#             writer.add_scalar('epoch_loss', running_loss / len(trainloader), epoch)
#             writer.add_scalar('epoch_accuracy', 100. * correct / total, epoch)
#
#         # 学习率调度
#         scheduler.step()
#
#         # eval testset
#         # raw_accuracy, local_accuracy, sum_accuracy = eval(model, testloader, criterion, 'test', save_path, epoch)
#         #
#         # print(
#         #     'Test set: raw accuracy: {:.2f}%'.format(100. * raw_accuracy))
#         # print(
#         #     'Test set: local accuracy: {:.2f}%'.format(100. * local_accuracy))
#         # print(
#         #     'Test set: sum accuracy: {:.2f}%'.format(100. * sum_accuracy))
#
#         eval(model, testloader, criterion, 'test', save_path, epoch)
#
#         # save checkpoint
#         if (epoch % save_interval == 0) or (epoch == end_epoch):
#             print('Saving checkpoint')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'learning_rate': lr,
#             }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
#
#         # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
#         # and delete the redundant ones
#         checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
#         if len(checkpoint_list) == max_checkpoint_num + 1:
#             idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
#             min_idx = min(idx_list)
#             os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))


import glob
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from config import max_checkpoint_num, proposalN, eval_trainset, set
from utils.eval_model import eval


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes=200, feat_dim=768, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval,
          writer=None):
    best_sum_accuracy = 0  # 跟踪验证集的最佳 Sum 准确率
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print(f'\nTraining {epoch} epoch')  # 只打印一次epoch开始
        lr = next(iter(optimizer.param_groups))['lr']

        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(total=len(trainloader), ncols=80, dynamic_ncols=True, desc=f"Epoch {epoch}/{end_epoch}", unit="batch") as pbar:
            for i, data in enumerate(trainloader):
                if set == 'CUB':
                    images, labels, _, _ = data
                else:
                    images, labels = data
                images, labels = images.cuda(), labels.cuda()


                optimizer.zero_grad()
                raw_logits, local_logits = model(images, epoch, i, 'train')

                raw_loss = criterion(raw_logits, labels)
                local_loss = criterion(local_logits, labels)

                total_loss = raw_loss + local_loss
                total_loss.backward()

                optimizer.step()

                running_loss += total_loss.item()

                # 计算准确度
                _, predicted = torch.max(raw_logits, 1) # predicted: 从 raw_logits 中获取每个样本预测的类别（概率最大的那个）。
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条
                pbar.set_postfix(loss=running_loss / (i + 1), accuracy=100. * correct / total)
                pbar.update(1)

                # 每10个批次记录一次
                if writer is not None and i % 10 == 0:
                    writer.add_scalar('training_loss', running_loss / (i + 1), epoch * len(trainloader) + i)
                    writer.add_scalar('training_accuracy', 100. * correct / total, epoch * len(trainloader) + i)

                    # 记录输入图像（如果需要）
                    if i < 5:  # 每个epoch记录前5个批次的图像
                        writer.add_images('train_images', images, epoch * 100 + i)

        # 每个epoch结束后，记录损失和准确率
        if writer is not None:
            writer.add_scalar('epoch_loss', running_loss / len(trainloader), epoch)
            writer.add_scalar('epoch_accuracy', 100. * correct / total, epoch)

        # 学习率调度
        scheduler.step()

        # eval testset
        #eval(model, testloader, criterion, 'test', save_path, epoch)

        raw_weight, local_weight, sum_acc = eval(model, testloader, criterion, 'test', save_path, epoch, writer)

        # 更新和保存最佳模型
        if sum_acc > best_sum_accuracy:
            best_sum_accuracy = sum_acc
            print(f"New best sum accuracy: {best_sum_accuracy*100:.2f}%, saving best model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_sum_accuracy': best_sum_accuracy,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(save_path, '512OL-best.pth'))


        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, f'epoch{epoch}.pth'))

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, f'epoch{min_idx}.pth'))
