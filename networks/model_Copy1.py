import torch
from skimage import measure
from torch import nn
import torch.nn.functional as F
from networks.tiny_vit import *


class MainNet(nn.Module):
    def __init__(self, proposalN=None, num_classes=None, channels=None):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.model = tiny_vit_21m_512(pretrained=True, num_classes=num_classes)

        self.norm = nn.LayerNorm(576)
        self.pre_head = nn.Linear(576, 576)


    def forward(self, x, status='test', DEVICE='cuda'):
    # def forward(self, x, status=None, debug=False, save_path=None):
        #     def forward(self, x):
        #         writer = SummaryWriter('log')

        # # 仅用于调试：保存输入图像或中间特征图
        # if debug and save_path is not None:
        #     import torchvision
        #     import os
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torchvision.utils.save_image(x, save_path, normalize=True)

        out,  weights = self.model(x)

        #
        B, _ = out.shape

        # 另外一个多头注意力
        attn_weights_new = []
        attn_weights_new.append(weights[-2])
        attn_weights_new.append(weights[-1])
        attn_weights_new = torch.stack(attn_weights_new)
        #         print('attn_weights_new',attn_weights_new.shape)
        weights = torch.mean(attn_weights_new, dim=0)
        #         print('weights',weights.shape)

        weight_ori = weights
        weights = None
        # print(f"weight_ori.size(0): {weight_ori.size(0)}, B: {B}")
        # weight_ori = weight_ori.reshape((B,int(weight_ori.size(0)/B),18,49,49))
        weight_ori = weight_ori.reshape((B, int(weight_ori.size(0) / B), 18, 256, 256))
        # 注意力权重处理：选取最后两层的注意力权重，进行堆叠并求平均，得到 weights。然后将其重塑为特定形状的 weight_ori。
        #         print('weight_ori',weight_ori.shape)

        M = torch.randn(weight_ori.shape[0], 16, 16).cuda()  #掩码生成：遍历每个样本的注意力权重，对不同头的注意力权重进行累加，得到 v。
        # 然后对 v 求均值并调整形状，得到 patch。根据 patch 的均值乘以一个阈值（这里是 0.200）得到 a，将 patch 中大于 a 的元素置为 1，其余置为 0，生成掩码 M。
        for item in range(weight_ori.shape[0]):
            weight = weight_ori[item]
            #             print('weight',weight.shape)
            weight = weight.transpose(1, 0)

            #             weight = weight / weight.sum(dim=-1).unsqueeze(-1)

            j = torch.zeros(weight.size()).cuda()
            j[0] = weight[0]
            #             print(j[0].shape)
            for n in range(1, weight.size(0)):
                # 18个head逐一相加
                #                 j[n] = torch.matmul(weight[n], j[n - 1])
                j[n] = torch.add(weight[n], j[n - 1])

            v = j[-1]
            #             print('v', v.shape)
            #             patch = torch.mean(v, dim=1) / (torch.mean(v, dim=1).max())
            patch = torch.mean(v, dim=1)
            #             print('patch', patch.shape)
            patch = patch.view(1, 2, 2, 8, 8)
            #             print('patch', patch.shape)
            patch = patch.permute(0, 1, 3, 2, 4).contiguous().view(16, 16)
            #             print('patch', patch.shape)
            #             print(patch.flatten())
            #a = torch.mean(patch.flatten()) * 0.100  ### clipping parameter
            #a = torch.mean(patch.flatten()) * 0.150
            a = torch.mean(patch.flatten()) * 0.200          # 公式13
            #             print('a',a.shape)
            M[item] = (patch > a).float()
        #             print('m',M[item].shape)

        coordinates = []
        # 坐标提取：将掩码 M 转换为 numpy 数组，使用 scikit-image 库的 measure.label 和 measure.regionprops 函数找出最大连通区域的边界框，
        # 计算其左上角和右下角的坐标，并进行边界检查，确保坐标不小于 0。
        for i, m in enumerate(M):
            mask_np = m.cpu().numpy().reshape(16, 16)
            component_labels = measure.label(mask_np, connectivity=2)

            properties = measure.regionprops(component_labels)
            areas = []
            for prop in properties:
                areas.append(prop.area)
            max_idx = areas.index(max(areas))

            bbox = properties[max_idx]['bbox']
            x_lefttop = bbox[0] * 32 - 1
            y_lefttop = bbox[1] * 32 - 1
            x_rightlow = bbox[2] * 32 - 1
            y_rightlow = bbox[3] * 32 - 1
            # for image
            if x_lefttop < 0:
                x_lefttop = 0
            if y_lefttop < 0:
                y_lefttop = 0
            coordinate = [int(x_lefttop), int(y_lefttop), int(x_rightlow), int(y_rightlow)]
            coordinates.append(coordinate)

        coordinates = torch.tensor(coordinates)  # 局部图像提取和分类：将坐标转换为 torch.tensor，根据坐标从输入图像 x 中提取局部图像，并使用 F.interpolate 函数将其调整为 (512, 512) 的大小。
        # 将局部图像传入 self.model 进行二次分类，得到输出 out2。最后返回全局分类结果 out 和局部分类结果 out2。
        batch_size = len(coordinates)
        local_imgs = torch.zeros([B, 3, 512, 512]).cuda()  # [N, 3, 448, 448]
        # local_imgs = torch.zeros([B, 3, 448, 448])
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(512, 512),
                                                mode='bilinear', align_corners=True)
        out2, weights = self.model(local_imgs.detach(), key='wtq')



        return out, out2