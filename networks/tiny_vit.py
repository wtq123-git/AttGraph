# # --------------------------------------------------------
# # TinyViT Model Architecture
# # Copyright (c) 2022 Microsoft
# # Adapted from LeViT and Swin Transformer
# #   LeViT: (https://github.com/facebookresearch/levit)
# #   Swin: (https://github.com/microsoft/swin-transformer)
# # Build the TinyViT Model
# # --------------------------------------------------------
#
# import itertools
# import logging
# import math
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath as TimmDropPath, \
#     to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from typing import Tuple
# # from .CA import *
# from torch.nn import Linear, Dropout
# # from efficient_kan import KAN
# # from module.SLA import SimplifiedLinearAttention as slaatt
#
# weights = []
# hidden = []
#
#
# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, # a: 3 b: 48
#                  groups=1, bn_weight_init=1):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         bn = torch.nn.BatchNorm2d(b)  # 创建一个二维批归一化层，输出通道数为 b
#         torch.nn.init.constant_(bn.weight, bn_weight_init) # 将权重初始化为常数 bn_weight_init
#         torch.nn.init.constant_(bn.bias, 0) # 将偏置初始化为 0
#         self.add_module('bn', bn) # 将该 BatchNorm2d 层添加到模块中，命名为 'bn'
#
#     @torch.no_grad()
#     def fuse(self): # 将卷积层与批归一化层融合成一个新的卷积层
#         c, bn = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps) ** 0.5
#         w = c.weight * w[:, None, None, None]
#         b = bn.bias - bn.running_mean * bn.weight / \
#             (bn.running_var + bn.eps) ** 0.5
#         m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
#             0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
#                             groups=self.c.groups)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m
#
#
# class DropPath(TimmDropPath):
#     def __init__(self, drop_prob=None):
#         super().__init__(drop_prob=drop_prob)
#         self.drop_prob = drop_prob
#
#     def __repr__(self):
#         msg = super().__repr__()
#         msg += f'(drop_prob={self.drop_prob})'
#         return msg
#
#
# class PatchEmbed(nn.Module):# 将输入图像转换为一组嵌入向量的模块
#     def __init__(self, in_chans, embed_dim, resolution, activation):  # 3 96 448
#         super().__init__()
#         img_size: Tuple[int, int] = to_2tuple(resolution)  # img_size(448, 448)
#         self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)  # (112,112)
#         self.num_patches = self.patches_resolution[0] * \
#                            self.patches_resolution[1]  # 计算的是总共的图像块数 112*112
#         self.in_chans = in_chans    # 3
#         self.embed_dim = embed_dim  # 96
#
#         n = embed_dim
#         self.seq = nn.Sequential(  # 将输入的图像或特征图进行处理，提取特征并进行下采样。通过使用步幅 2 和卷积核 3x3，每个卷积层都会减小输入的空间分辨率，同时增加输出特征的维度。
#             # 创建一个序列容器，按顺序执行多个层操作
#             Conv2d_BN(in_chans, n // 2, 3, 2, 1),  # 图片大小448/2 = 224  输出通道数 96/2 = 48
#             activation(),
#             Conv2d_BN(n // 2, n, 3, 2, 1),   # 图片大小224/2 = 112  通道数 96
#         )
#
#     def forward(self, x):
#         #print("!!", x.shape)
#         return self.seq(x)  # x 输出尺寸：(96, 112, 112)
#
#
# class MBConv(nn.Module):
#     # MBConv（Mobile Inverted Bottleneck Convolution）是一个常用于轻量级模型（如 MobileNetV2 和 EfficientNet）中的模块，
#     # 尤其是在设计高效卷积神经网络时。它包含了瓶颈层、深度可分离卷积和线性瓶颈等操作。
#     # 通过将卷积操作与扩展比率和激活函数相结合，MBConv 模块能够在保持模型效率的同时提取更丰富的特征。
#     def __init__(self, in_chans, out_chans, expand_ratio,  # 96 96
#                  activation, drop_path):
#         super().__init__()
#         self.in_chans = in_chans
#         self.hidden_chans = int(in_chans * expand_ratio)
#         self.out_chans = out_chans
#
#         self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)   # 96 384
#         self.act1 = activation()
#
#         self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
#                                ks=3, stride=1, pad=1, groups=self.hidden_chans)  #  384 384
#         self.act2 = activation()
#
#         self.conv3 = Conv2d_BN(
#             self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)  # 384 96
#         self.act3 = activation()
#
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#
#     #         self.ca1 = CoordAtt(384,384)
#     #         self.ca2 = CoordAtt(384,384)
#     #         self.ca3 = CoordAtt(96,96)
#
#     def forward(self, x):
#         shortcut = x
#
#         x = self.conv1(x)
#         x = self.act1(x)
#
#         x = self.conv2(x)
#         x = self.act2(x)
#
#         x = self.conv3(x)
#         #         x = self.ca3(x)
#
#         x = self.drop_path(x)
#
#         x += shortcut
#         x = self.act3(x)
#
#         return x
#
#
# class PatchMerging(nn.Module):
#     def __init__(self, input_resolution, dim, out_dim, activation):
#         super().__init__()
#
#         self.input_resolution = input_resolution  # （112，112）
#         self.dim = dim
#         self.out_dim = out_dim  # 192
#         self.act = activation()
#         self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)    # 96 192
#         self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)  # 192 192
#         self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)
#
#     def forward(self, x):
#         if x.ndim == 3:
#             H, W = self.input_resolution
#             B = len(x)
#             # print("@@", B,H,W)
#             # (B, C, H, W)
#             x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
#
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.conv2(x)
#         x = self.act(x)
#         x = self.conv3(x)
#
#         x = x.flatten(2).transpose(1, 2)
#         return x
#
#
# class ConvLayer(nn.Module):
#     def __init__(self, dim, input_resolution, depth,
#                  activation,
#                  drop_path=0., downsample=None, use_checkpoint=False,
#                  out_dim=None, # 192
#                  conv_expand_ratio=4.,
#                  ):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#
#         # build blocks
#         self.blocks = nn.ModuleList([  # 用于存储 depth 个 MBConv 块
#             MBConv(dim, dim, conv_expand_ratio, activation,
#                    drop_path[i] if isinstance(drop_path, list) else drop_path,
#                    )
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(
#                 input_resolution, dim=dim, out_dim=out_dim, activation=activation)
#         else:
#             self.downsample = None
#
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None,
#                  out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.norm = nn.LayerNorm(in_features)
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.act = act_layer()
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.norm(x)
#
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class Attention(torch.nn.Module):
#     def __init__(self, dim, key_dim, num_heads=8, # dim: 192 key_dim: 32 num_heads: 6
#                  attn_ratio=4,
#                  resolution=(14, 14), # (7,7)
#                  ):
#         super().__init__()
#         # (h, w)
#         assert isinstance(resolution, tuple) and len(resolution) == 2
#         # 用于确保 resolution 是一个二元元组，通常用于验证图像分辨率或其他类似的二维数据格式。
#         self.num_heads = num_heads
#         self.scale = key_dim ** -0.5 # 计算了自注意力机制中的 缩放因子，通过对 key_dim 进行平方根倒数运算，目的是在计算查询和键的点积时避免数值过大或过小，保证模型的训练更加稳定。
#         self.key_dim = key_dim
#         self.nh_kd = nh_kd = key_dim * num_heads # 192
#         self.d = int(attn_ratio * key_dim)
#         self.dh = int(attn_ratio * key_dim) * num_heads # 1*32*6 = 192
#         self.attn_ratio = attn_ratio  # 1
#         h = self.dh + nh_kd * 2  # h: 576
#
#         self.norm = nn.LayerNorm(dim)  # 用于对输入进行标准化
#         self.qkv = nn.Linear(dim, h)    # 用于通过线性层计算查询、键和值
#         self.proj = nn.Linear(self.dh, dim) # 用于将多个注意力头的输出映射回原始的 dim 维度
#
#         points = list(itertools.product(
#             range(resolution[0]), range(resolution[1])))
#         # itertools.product(range(resolution[0]), range(resolution[1])) 生成一个二维网格的所有可能的点对。
#         # points 列表包含了所有这些点，N 是这些点的总数。
#         N = len(points) # 7*7 = 49
#         attention_offsets = {}
#         idxs = []
#         for p1 in points:
#             for p2 in points:
#                 offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
#                 if offset not in attention_offsets:
#                     attention_offsets[offset] = len(attention_offsets)
#                 idxs.append(attention_offsets[offset])
#         self.attention_biases = torch.nn.Parameter(
#             torch.zeros(num_heads, len(attention_offsets)))
#         self.register_buffer('attention_bias_idxs',
#                              torch.LongTensor(idxs).view(N, N),
#                              persistent=False)
#         # 将 idxs 转换为一个形状为 (N, N) 的 LongTensor，表示每个点对之间的偏移量索引，并将其注册为一个缓冲区（register_buffer）以便后续使用。
#
#     @torch.no_grad()
#     def train(self, mode=True):
#         super().train(mode)
#         if mode and hasattr(self, 'ab'):
#             del self.ab
#         else:
#             self.ab = self.attention_biases[:, self.attention_bias_idxs]
#
#     def forward(self, x):  # x (B,N,C)
#         B, N, _ = x.shape
#
#         # Normalization
#         x = self.norm(x)
#
#         qkv = self.qkv(x)
#         # (B, N, num_heads, d)
#         q, k, v = qkv.view(B, N, self.num_heads, -
#         1).split([self.key_dim, self.key_dim, self.d], dim=3)
#         # (B, num_heads, N, d)
#         q = q.permute(0, 2, 1, 3)
#         k = k.permute(0, 2, 1, 3)
#         v = v.permute(0, 2, 1, 3)
#
#         attn = (
#                 (q @ k.transpose(-2, -1)) * self.scale
#                 +
#                 (self.attention_biases[:, self.attention_bias_idxs]
#                  if self.training else self.ab)
#         )
#
#         attn_temp = attn
#         attn = attn.softmax(dim=-1)
#
#         #         print(attn.shape)
#         global weights
#         if len(weights) == 12:
#             weights = []
#
#         weights.append(attn)
#         # weights.append(attn_temp)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
#         x = self.proj(x)
#         #         print(x.shape)
#         return x
#
#
# class TinyViTBlock(nn.Module):
#     r""" TinyViT Block.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int, int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         drop (float, optional): Dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         local_conv_size (int): the kernel size of the convolution between
#                                Attention and MLP. Default: 3
#         activation: the activation function. Default: nn.GELU
#     """
#
#     def __init__(self, dim, input_resolution, num_heads, window_size=7,
#                  mlp_ratio=4., drop=0., drop_path=0.,
#                  local_conv_size=3,
#                  activation=nn.GELU,
#                  ):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         assert window_size > 0, 'window_size must be greater than 0'
#         self.window_size = window_size
#         self.mlp_ratio = mlp_ratio
#
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#
#         assert dim % num_heads == 0, 'dim must be divisible by num_heads'
#         head_dim = dim // num_heads  # 32
#
#         window_resolution = (window_size, window_size) # (7,7)
#         self.attn = Attention(dim, head_dim, num_heads,
#                               attn_ratio=1, resolution=window_resolution)
#
#         mlp_hidden_dim = int(dim * mlp_ratio)   # 192*4= 768
#         mlp_activation = activation
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
#                        act_layer=mlp_activation, drop=drop)
#
#         pad = local_conv_size // 2
#         self.local_conv = Conv2d_BN(
#             dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)
#
#     def forward(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         # print("!!",B,L,C,H,W)
#         assert L == H * W, "input feature has wrong size"
#         res_x = x
#         if H == self.window_size and W == self.window_size:
#             x = self.attn(x)
#         else:
#             x = x.view(B, H, W, C)
#             pad_b = (self.window_size - H %
#                      self.window_size) % self.window_size
#             pad_r = (self.window_size - W %
#                      self.window_size) % self.window_size
#             padding = pad_b > 0 or pad_r > 0
#
#             if padding:
#                 x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
#
#             pH, pW = H + pad_b, W + pad_r
#             nH = pH // self.window_size
#             nW = pW // self.window_size
#             # window partition
#             x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
#                 B * nH * nW, self.window_size * self.window_size, C
#             )
#             x = self.attn(x)
#             # window reverse
#             x = x.view(B, nH, nW, self.window_size, self.window_size,
#                        C).transpose(2, 3).reshape(B, pH, pW, C)
#
#             if padding:
#                 x = x[:, :H, :W].contiguous()
#
#             x = x.view(B, L, C)
#
#         x = res_x + self.drop_path(x)
#
#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         x = self.local_conv(x)
#         x = x.view(B, C, L).transpose(1, 2)
#
#         x = x + self.drop_path(self.mlp(x))
#         #         print(x.shape)
#         global hidden
#         if len(hidden) == 10:
#             hidden = []
#         hidden.append(x)
#
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
#
#
# class BasicLayer(nn.Module):
#     # BasicLayer 用于构建一个包含多个 TinyViTBlock 的网络层，TinyViTBlock 是 TinyViT 中的基本计算单元
#     """ A basic TinyViT layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         drop (float, optional): Dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#         local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
#         activation: the activation function. Default: nn.GELU
#         out_dim: the output dimension of the layer. Default: dim
#     """
#
#     def __init__(self, dim, input_resolution, depth, num_heads, window_size, # dim 192 num_heads 6
#                  mlp_ratio=4., drop=0.,
#                  drop_path=0., downsample=None, use_checkpoint=False,
#                  local_conv_size=3,
#                  activation=nn.GELU,
#                  out_dim=None,
#                  ):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth  # 层内包含的 TinyViTBlock 的数量，即有多少个计算单元（深度）
#         self.use_checkpoint = use_checkpoint
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             TinyViTBlock(dim=dim, input_resolution=input_resolution,
#                          num_heads=num_heads, window_size=window_size,
#                          mlp_ratio=mlp_ratio,
#                          drop=drop,
#                          drop_path=drop_path[i] if isinstance(
#                              drop_path, list) else drop_path,
#                          local_conv_size=local_conv_size,
#                          activation=activation,
#                          )
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(
#                 input_resolution, dim=dim, out_dim=out_dim, activation=activation)
#         else:
#             self.downsample = None
#
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#
# class TinyViT(nn.Module):
#     def __init__(self, img_size=448, in_chans=3, num_classes=2884,
#                  embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
#                  num_heads=[3, 6, 12, 24],
#                  window_sizes=[7, 7, 14, 7],
#                  mlp_ratio=4.,
#                  drop_rate=0.,
#                  drop_path_rate=0.1,
#                  use_checkpoint=False,
#                  # use_checkpoint=False 表示 不启用梯度检查点，即每个层的中间激活值都将被存储，适合于显存足够的情况。
#                  # use_checkpoint=True 表示启用梯度检查点，可以在显存有限的情况下减少内存占用，但会增加计算开销。
#                  mbconv_expand_ratio=4.0,
#                  local_conv_size=3,
#                  layer_lr_decay=1.0,
#                  ):
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.depths = depths
#         self.num_layers = len(depths) # 4
#         self.mlp_ratio = mlp_ratio
#
#         activation = nn.GELU
#
#         self.patch_embed = PatchEmbed(in_chans=in_chans,
#                                       embed_dim=embed_dims[0], # 96
#                                       resolution=img_size,
#                                       activation=activation) # x 输出尺寸：(96, 112, 112)
#
#         patches_resolution = self.patch_embed.patches_resolution # (112, 112)
#         self.patches_resolution = patches_resolution
#
#         # stochastic depth  随机深度技术目的是在训练过程中对每一层应用一定的概率性“丢弃”操作，从而让网络具有更好的泛化能力。
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
#                                                 sum(depths))]
#
#         # torch.linspace(start, end, steps) 生成一个在 start 和 end 之间均匀分布的指定数量（steps）的值
#         # drop_path_rate 最终希望达到的丢弃率
#         # DropPath 是一种用于网络训练的技术，它类似于 Dropout，但是与 Dropout 不同的是，它是在网络的“路径”级别进行丢弃，
#         # 而不是单独的神经元。在 DropPath 中，一些路径（即网络层）在训练过程中被随机丢弃，降低模型对特定路径的依赖，从而增加模型的泛化能力。
#         # depths：这个列表通常包含了网络的每一层或每一模块的深度（或模块的数量），每个模块会对应一定数量的层。因此，sum(depths) =  12 就是网络中层的总数。
#
#         # build layers
#         self.layers = nn.ModuleList() # 创建了一个空的 ModuleList，用于存储各个层
#         for i_layer in range(self.num_layers):  #  num_layers = 4
#             kwargs = dict(dim=embed_dims[i_layer],  # kwargs 字典用于存储本次循环中每一层的初始化参数
#                           input_resolution=(patches_resolution[0] // (2 ** i_layer),
#                                             patches_resolution[1] // (2 ** i_layer)),
#                           depth=depths[i_layer],
#                           drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                           downsample=PatchMerging if (
#                                   i_layer < self.num_layers - 1) else None,
#                           use_checkpoint=use_checkpoint,
#                           out_dim=embed_dims[min(
#                               i_layer + 1, len(embed_dims) - 1)], # 192
#                           activation=activation,
#                           )
#             if i_layer == 0:
#                 layer = ConvLayer(
#                     conv_expand_ratio=mbconv_expand_ratio,  # 4.0
#                     **kwargs,
#                 )
#             else:
#                 layer = BasicLayer(
#                     num_heads=num_heads[i_layer],
#                     window_size=window_sizes[i_layer],
#                     mlp_ratio=self.mlp_ratio,
#                     drop=drop_rate,
#                     local_conv_size=local_conv_size,
#                     **kwargs)
#             self.layers.append(layer)
#
#         # Classifier head
#         self.norm_head = nn.LayerNorm(576)
#         self.norm_head_local = nn.LayerNorm(576)   # norm_head 和 norm_head_local 是两个分别用于 raw logits 和 local logits 的归一化层
#
#         # Set the output output dimension according to your situation
#         self.head = nn.Linear(
#             576, num_classes) if num_classes > 0 else torch.nn.Identity()
#         self.head_local = nn.Linear(
#             576, num_classes) if num_classes > 0 else torch.nn.Identity()
#         # raw
#         self.head_pre = nn.Linear(576 + 576   # 这个层的作用是将两个 576 维的输入融合，并将其映射到一个 576 维的输出。
#             , 576) if num_classes > 0 else torch.nn.Identity()
#         # local
#         self.head_pre_local = nn.Linear(576 + 576
#             , 576) if num_classes > 0 else torch.nn.Identity()
#
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#
#         self.norm_head_1 = nn.LayerNorm(576)
#         self.norm_head_1_local = nn.LayerNorm(576)
#
#         self.norm_head_2 = nn.LayerNorm(576)
#         self.norm_head_2_local = nn.LayerNorm(576)
#
#
#         self.part_structure_local = Part_Structure() # SRFL 模块
#         self.part_structure_raw = Part_Structure()
#
#         # init weights
#         self.apply(self._init_weights)
#         self.set_layer_lr_decay(layer_lr_decay)
#
#         # SLA
#         # self.slaatt = slaatt(3, [448, 448], 1)
#
#     def set_layer_lr_decay(self, layer_lr_decay):
#         decay_rate = layer_lr_decay
#
#         # layers -> blocks (depth)
#         depth = sum(self.depths)
#         lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
#         print("LR SCALES:", lr_scales)
#
#         def _set_lr_scale(m, scale):
#             for p in m.parameters():
#                 p.lr_scale = scale
#
#         self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
#         i = 0
#         for layer in self.layers:
#             for block in layer.blocks:
#                 block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
#                 i += 1
#             if layer.downsample is not None:
#                 layer.downsample.apply(
#                     lambda x: _set_lr_scale(x, lr_scales[i - 1]))
#         assert i == depth
#         for m in [self.norm_head, self.head]:
#             m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))
#
#         for k, p in self.named_parameters():
#             p.param_name = k
#
#
#
#     #         def _check_lr_scale(m):
#     #             for p in m.parameters():
#     #                 assert hasattr(p, 'lr_scale'), p.param_name
#
#     #         self.apply(_check_lr_scale)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'attention_biases'}
#
#
#
#     def forward_features(self, x, key=None):
#         x = self.patch_embed(x)
#
#         x = self.layers[0](x)
#         start_i = 1
#
#         for i in range(start_i, len(self.layers)):
#             layer = self.layers[i]
#             x = layer(x)
#
#         global weights, hidden
#
#         attn_weights_new = []
#         attn_weights_new.append(weights[-2])
#         attn_weights_new.append(weights[-1])
#         attn_weights_new = torch.stack(attn_weights_new)
#         weights_A = torch.mean(attn_weights_new, dim=0)
#         # 从 weights 中提取出最后两层的注意力权重，然后计算它们的均值，得到一个加权的特征表示 weights_A
#
#         B, a, b, c = weights[-1].shape
#         B = int(B / 4)
#         weight = weights_A.view(B, 4, a, b, c)
#
#         patch = torch.mean(weight, dim=3)
#         patch = patch.view(B, 18, 2, 2, 7, 7)
#         patch = patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, 18, 14, 14)
#
#
#         # x  14,14,576
#
#         if key == None:
#             x = x.mean(1)
#             x = self.norm_head(x)
#             hidden_states = self.part_structure_raw(hidden[-1], patch)
#             hidden_states = self.norm_head_1(hidden_states)
#             # print(f"x shape: {x.shape}, hidden_states shape: {hidden_states.shape}")
#             x = torch.cat((x, hidden_states), dim=-1)
#             x = self.head_pre(x)
#             x = self.norm_head_2(x)
#         else:
#             x = x.mean(1)
#             x = self.norm_head_local(x)
#             hidden_states = self.part_structure_local(hidden[-1], patch)
#             hidden_states = self.norm_head_1_local(hidden_states)
#             # print(f"x shape: {x.shape}, hidden_states shape: {hidden_states.shape}")
#             x = torch.cat((x, hidden_states), dim=-1)
#             x = self.head_pre_local(x)
#             x = self.norm_head_2_local(x)
#
#         return x
#
#     def forward(self, x,key=None):
#         # print("!!!", x.shape)
#         global weights
#         # x = self.slaatt(x)
#         x = self.forward_features(x,key)
#         if key == None:
#             x = self.head(x)
#         else:
#             x = self.head_local(x)
#         return x,weights
#
#
# _checkpoint_url_format = \
#     'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'
# _provided_checkpoints = {
#     'tiny_vit_5m_224': 'tiny_vit_5m_22kto1k_distill',
#     'tiny_vit_11m_224': 'tiny_vit_11m_22kto1k_distill',
#     'tiny_vit_21m_224': 'tiny_vit_21m_22k_distill',
#     'tiny_vit_21m_384': 'tiny_vit_21m_22kto1k_384_distill',
#     'tiny_vit_21m_512': 'tiny_vit_21m_22kto1k_512_distill',
# }
#
#
# def register_tiny_vit_model(fn):
#     '''Register a TinyViT model
#     It is a wrapper of `register_model` with loading the pretrained checkpoint.
#     '''
#
#     def fn_wrapper(pretrained=False, **kwargs):
#         model = fn()
#         if pretrained:
#
#             # Replace pre-trained's model path according to your situation
#             # You can download in '_checkpoint_url_format' above
#             path = ('/project/tuchaohu/wtq/SRFL_OL1/tiny_vit_21m_22kto1k_512_distill.pth')
#             checkpoint = torch.load(path,
#                                     map_location='cpu')
#
#             net_dict = model.state_dict()
#             # predict_model = torch.load(pth_path)
#             # state_dict = {k: v for k, v in checkpoint["model"].items() if k in net_dict.keys()}
#             state_dict = {k: v for k, v in checkpoint["model"].items() if
#                           k in net_dict.keys() and 'head' not in k and 'part_structure' not in k and 'part_head' not in k}
#             net_dict.update(state_dict)
#
#             model.load_state_dict(net_dict)
#
#         return model
#
#     # rename the name of fn_wrapper
#     fn_wrapper.__name__ = fn.__name__
#     return register_model(fn_wrapper)
#
#
# @register_tiny_vit_model
# def tiny_vit_5m_224(pretrained=False, num_classes=2884, drop_path_rate=0.0):
#     return TinyViT(
#         num_classes=num_classes,
#         embed_dims=[64, 128, 160, 320],
#         depths=[2, 2, 6, 2],
#         num_heads=[2, 4, 5, 10],
#         window_sizes=[7, 7, 14, 7],
#         drop_path_rate=drop_path_rate,
#     )
#
#
# @register_tiny_vit_model
# def tiny_vit_11m_224(pretrained=False, num_classes=1000, drop_path_rate=0.1):
#     return TinyViT(
#         num_classes=num_classes,
#         embed_dims=[64, 128, 256, 448],
#         depths=[2, 2, 6, 2],
#         num_heads=[2, 4, 8, 14],
#         window_sizes=[7, 7, 14, 7],
#         drop_path_rate=drop_path_rate,
#     )
#
#
# @register_tiny_vit_model
# def tiny_vit_21m_224(pretrained=False, num_classes=2884, drop_path_rate=0.2):
#     return TinyViT(
#         img_size=448,
#         num_classes=num_classes,
#         embed_dims=[96, 192, 384, 576],
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 18],
#         window_sizes=[7, 7, 14, 7],
#         drop_path_rate=drop_path_rate,
#     )
#
#
# @register_tiny_vit_model
# def tiny_vit_21m_384(pretrained=False, num_classes=1000, drop_path_rate=0.1):
#     return TinyViT(
#         img_size=384,
#         num_classes=num_classes,
#         embed_dims=[96, 192, 384, 576],
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 18],
#         window_sizes=[12, 12, 24, 12],
#         drop_path_rate=drop_path_rate,
#     )
#
#
# @register_tiny_vit_model
# def tiny_vit_21m_512(pretrained=False, num_classes=2884, drop_path_rate=0.1):
#     return TinyViT(
#         img_size=512,
#         num_classes=num_classes,
#         embed_dims=[96, 192, 384, 576],
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 18],
#         window_sizes=[16, 16, 32, 16],
#         drop_path_rate=drop_path_rate,
#     )
#
#
# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, bias=True, dropout=0.1):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.zeros(in_features, out_features))
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.relu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(p=dropout)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         weight = self.weight.float()
#         support = torch.matmul(input, weight.cuda())
#         output = torch.matmul(adj, support)
#         if self.bias is not None:
#             return self.dropout(self.relu(output + self.bias))
#         else:
#             return self.dropout(self.relu(output))
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
#
#
# '''Laplacian Matrix transorm'''
#
#
# def gen_adj(A):
#     D = torch.pow(A.sum(1).float(), -0.5)
#     D = torch.diag(D)
#     adj = torch.matmul(torch.matmul(A, D).t(), D)
#     return adj
#
#
# logger = logging.getLogger(__name__)
#
# ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
# ATTENTION_K = "MultiHeadDotProductAttention_1/key"
# ATTENTION_V = "MultiHeadDotProductAttention_1/value"
# ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
# FC_0 = "MlpBlock_3/Dense_0"
# FC_1 = "MlpBlock_3/Dense_1"
# ATTENTION_NORM = "LayerNorm_0"
# MLP_NORM = "LayerNorm_2"
#
#
# def np2th(weights, conv=False):
#     """Possibly convert HWIO to OIHW."""
#     if conv:
#         weights = weights.transpose([3, 2, 0, 1])
#     return torch.from_numpy(weights)
#
#
# def swish(x):
#     return x * torch.sigmoid(x)
#
#
# ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
#
#
# class RelativeCoordPredictor(nn.Module): # 用于计算输入的 attention_map 中相对坐标的信息
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         N, C, H, W = x.shape
#         # print("00", x.shape)
#         # reduced_x = x.view(N, C, H*W).transpose(1, 2).contiguous()  # (N, S, C)
#
#         mask = torch.sum(x, dim=1)
#         size = H
#
#         # mask = mask.view(N, H * W)
#         # thresholds = torch.mean(mask, dim=1, keepdim=True)
#         # binary_mask = (mask > (thresholds*0.25)).float()
#         # binary_mask = binary_mask.view(N, H, W)
#
#         binary_mask = torch.ones(N, H, W).cuda()
#
#         masked_x = x * binary_mask.view(N, 1, H, W)
#         masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
#         _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)
#
#         basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()
#         #         basic_index = torch.from_numpy(np.array([i for i in range(N)]))
#
#         # max_features = reduced_x[basic_index, reduced_x_max_index, :]  # (N, C)
#         # max_features_to_concat = max_features.unsqueeze(1).expand((N, H*W, C))
#
#         basic_label = torch.from_numpy(self.build_basic_label(size)).float()
#         # Build Label
#         label = basic_label.cuda()
#         #         label = basic_label
#
#         label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)
#         # label = label.type(torch.long)
#         basic_anchor = label[basic_index.long(), reduced_x_max_index.long(), :].unsqueeze(1)  # (N, 1, 2)
#         relative_coord = label - basic_anchor
#         relative_coord = relative_coord / size
#         relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)
#         relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
#         relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)
#
#         binary_relative_mask = binary_mask.view(N, H * W)
#         relative_dist = relative_dist * binary_relative_mask
#         relative_angle = relative_angle * binary_relative_mask
#
#         basic_anchor = basic_anchor.squeeze(1)  # (N, 2)
#
#         relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)
#
#         '''# Calc Loss
#         preds_dist, preds_angle = preds_coord[:, :, :, 0], preds_coord[:, :, :, 1]
#
#         preds_dist = preds_dist.view(N, H, W)
#         relative_dist = relative_dist.view(N, H, W)
#         dist_loss = self.dist_loss_f(preds_dist, relative_dist)
#
#         preds_angle = preds_angle.view(N, H*W)
#         gap_angle = preds_angle - relative_angle  # (N, S) in (0, 1) - (0, 1) = (-1, 1)
#         gap_angle[gap_angle < 0] += 1
#         gap_angle = gap_angle - torch.mean(gap_angle, dim=-1, keepdim=True)  # (N, H*W)
#         gap_angle = gap_angle.view(N, H, W)
#         angle_loss = torch.pow(gap_angle, 2)
#         '''
#
#         position_weight = torch.mean(masked_x, dim=-1)
#         position_weight = position_weight.unsqueeze(2)
#         position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))
#
#         # relative_coord_total = relative_coord_total.half()
#         # position_weight = position_weight.half()
#
#         return relative_coord_total, basic_anchor, position_weight, reduced_x_max_index
#
#     def build_basic_label(self, size):
#         basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
#         return basic_label
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout)
#         x = self.gc2(x, adj)
#         return x
#
#
# class Part_Structure(nn.Module):
#     # 输入的 attention_map 中提取结构信息，结合图卷积网络（GCN）对结构信息进行建模，并最终生成一个形状为 (B, 576) 的特征张量，作为网络的输出。
#     def __init__(self, config=None):
#         super(Part_Structure, self).__init__()
#
#         ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
#         self.act_fn = ACT2FN["relu"]
#         self.dropout = Dropout(0.1)
#         self.relative_coord_predictor = RelativeCoordPredictor()  # 位置坐标预测模块
#
#         self.gcn = GCN(2, 2, 576, dropout=0.1)  # 图卷积网络，用于处理结构信息
#
#     def forward(self, hidden_states, attention_map):
#         B, C, H, W = attention_map.shape
#         structure_info, basic_anchor, position_weight, reduced_x_max_index = self.relative_coord_predictor(
#             attention_map)
#
#         structure_info = self.gcn(structure_info, position_weight)
#         # GCN 通过考虑节点（图中的每个局部区域）的连接和邻居关系来进行卷积操作。这里 structure_info 和 position_weight 被用作 GCN 的输入，更新了 structure_info。
#
#         out = torch.ones((B, 576)).cuda()
#         for i in range(B):
#             index = int(basic_anchor[i, 0] * H + basic_anchor[i, 1])
#
#             # hidden_states[i, 0] = hidden_states[i, 0] + structure_info[i, index, :]
#
#             out[i] = structure_info[i, index, :]
#
#         # return structure_info
#         return out # (B, 576)
#
# # net = tiny_vit_21m_224(pretrained=False)
# # input = torch.ones((2,3,448,448))
# # net(input)

# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath, \
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple
# from .CA import *
from torch.nn import Linear, Dropout
from visualize_structure import visualize_relative_info
# from module.SLA import SimplifiedLinearAttention as slaatt

weights = []
hidden = []


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
                           self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            # (B, 48, 256, 256)
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
            # (B, 96, 128, 128)
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    #         self.ca1 = CoordAtt(384,384)
    #         self.ca2 = CoordAtt(384,384)
    #         self.ca3 = CoordAtt(96,96)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        #         x = self.ca3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # print("@@", B, H, W)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        attn_temp = attn
        attn = attn.softmax(dim=-1)

        #         print(attn.shape)
        global weights
        if len(weights) == 12:
            weights = []

        weights.append(attn)
        # weights.append(attn_temp)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        #         print(x.shape)
        return x


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        # print("!!", B, L, C, H, W)
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        #         print(x.shape)
        global hidden
        if len(hidden) == 10:
            hidden = []
        hidden.append(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(nn.Module):
    def __init__(self, img_size=512, in_chans=3, num_classes=2884,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)  ## x 输出尺寸：(B, 96, 128, 128)

        patches_resolution = self.patch_embed.patches_resolution # 128,128
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # torch.linspace(0, drop_path_rate, sum(depths)) 生成一个从 0 到 drop_path_rate 的等差数列，元素个数为 sum(depths)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(int(patches_resolution[0] // (2 ** i_layer)),
                                            int(patches_resolution[1] // (2 ** i_layer))),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                                  i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head  分类头部分：包括多个归一化层和线性层，用于将特征映射到类别数。
        self.norm_head = nn.LayerNorm(576)
        self.norm_head_local = nn.LayerNorm(576)

        # Set the output output dimension according to your situation
        self.head = nn.Linear(
            576, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.head_local = nn.Linear(
            576, num_classes) if num_classes > 0 else torch.nn.Identity()

        self.head_pre = nn.Linear(576 + 576
            , 576) if num_classes > 0 else torch.nn.Identity()
        self.head_pre_local = nn.Linear(576 + 576
            , 576) if num_classes > 0 else torch.nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.norm_head_1 = nn.LayerNorm(576)
        self.norm_head_1_local = nn.LayerNorm(576)

        self.norm_head_2 = nn.LayerNorm(576)
        self.norm_head_2_local = nn.LayerNorm(576)


        self.part_structure_local = Part_Structure()
        self.part_structure_raw = Part_Structure()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

        #self.slaatt = slaatt(3, [448, 448], 1 )

    def set_layer_lr_decay(self, layer_lr_decay): # 通过分层学习率衰减的方式，为模型的不同层设置不同的学习率缩放因子，有助于在训练过程中对不同层的参数进行更精细的控制。
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k



    #         def _check_lr_scale(m):
    #             for p in m.parameters():
    #                 assert hasattr(p, 'lr_scale'), p.param_name

    #         self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x, key=None):
    # def forward_features(self, x, key=None, debug=False, save_path="debug_viz.png"):
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)

        global weights, hidden

        attn_weights_new = []
        attn_weights_new.append(weights[-2])
        attn_weights_new.append(weights[-1])
        #print(f"weights[-2] shape: {weights[-2].shape}")
        #print(f"weights[-1] shape: {weights[-1].shape}")

        attn_weights_new = torch.stack(attn_weights_new)
        weights_A = torch.mean(attn_weights_new, dim=0)    # 公式 5
        #print(f"weights_A shape: {weights_A.shape}")

        B, a, b, c = weights[-1].shape
        #print(f"B before division: {B}")
        #B = int(B / 4)
        # B = B // 4
        #print("!!", B)
        weight = weights_A.view(B, a, b, c )
        #print("@@@",weight.shape)

        patch = torch.mean(weight, dim=3)
        # print("@@@", patch.shape)
        # patch = patch.view(B, 18, 2, 2, 7, 7)
        # patch = patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, 18, 14, 14)
        patch = patch.view(B, 18, 16, 16)
        #patch = patch.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, 18, 32, 32)


        # x  14,14,576

        if key == None:
            x = x.mean(1)
            x = self.norm_head(x)
            hidden_states = self.part_structure_raw(hidden[-1], patch)
            # hidden_states = self.part_structure_raw(hidden[-1], patch, debug=debug, save_path=save_path)

            hidden_states = self.norm_head_1(hidden_states)
            # 扩展 hidden_states 以匹配 x 的 batch size
            hidden_states = hidden_states.repeat(x.size(0) // hidden_states.size(0), 1)



            # print(f"x shape: {x.shape}, hidden_states shape: {hidden_states.shape}")

            x = torch.cat((x, hidden_states), dim=-1)  # 进行 conact
            x = self.head_pre(x)
            x = self.norm_head_2(x)
        else:
            x = x.mean(1)
            x = self.norm_head_local(x)
            hidden_states = self.part_structure_local(hidden[-1], patch)
            #hidden_states = self.part_structure_local(hidden[-1], patch, debug=debug, save_path=save_path)

            hidden_states = self.norm_head_1_local(hidden_states)
            # 扩展 hidden_states 以匹配 x 的 batch size
            hidden_states = hidden_states.repeat(x.size(0) // hidden_states.size(0), 1)

            x = torch.cat((x, hidden_states), dim=-1)
            x = self.head_pre_local(x)
            x = self.norm_head_2_local(x)

        return x   # forward_features 函数通过一系列的操作对输入特征进行处理，并根据 key 的值选择不同的处理路径，最终输出处理后的特征。

    #def forward(self, x, key=None, debug=False, save_path="debug_viz.png"):
    def forward(self, x, key=None):
        # print("!!!", x.shape)  # torch.Size([8, 3, 448, 448])
        global weights
        # x = self.slaatt(x)  # SLA注意力
        # x = self.forward_features(x, key=key, debug=debug, save_path=save_path)
        x = self.forward_features(x, key)
        if key == None:
            x = self.head(x)  # 线性层，用于将特征映射到类别数
        else:
            x = self.head_local(x)
        return x, weights  # 返回分类结果和注意力权重
        # x :( B, num_class)
        # weights : 注意力权重列表



_checkpoint_url_format = \
    'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'
_provided_checkpoints = {
    'tiny_vit_5m_224': 'tiny_vit_5m_22kto1k_distill',
    'tiny_vit_11m_224': 'tiny_vit_11m_22kto1k_distill',
    'tiny_vit_21m_224': 'tiny_vit_21m_22k_distill',
    'tiny_vit_21m_384': 'tiny_vit_21m_22kto1k_384_distill',
    'tiny_vit_21m_512': 'tiny_vit_21m_22kto1k_512_distill',
}


def register_tiny_vit_model(fn):
    '''Register a TinyViT model
    It is a wrapper of `register_model` with loading the pretrained checkpoint.
    '''

    def fn_wrapper(pretrained=False, **kwargs):
        model = fn()
        if pretrained:

            # Replace pre-trained's model path according to your situation
            # You can download in '_checkpoint_url_format' above

            path = r'F:\paper\AttGraph\tiny_vit_21m_22kto1k_512_distill.pth'

            checkpoint = torch.load(path,
                                    map_location='cpu')

            net_dict = model.state_dict()
            # predict_model = torch.load(pth_path)
            # state_dict = {k: v for k, v in checkpoint["model"].items() if k in net_dict.keys()}
            state_dict = {k: v for k, v in checkpoint["model"].items() if
                          k in net_dict.keys() and 'head' not in k and 'part_structure' not in k and 'part_head' not in k}
            net_dict.update(state_dict)

            model.load_state_dict(net_dict)

        return model

    # rename the name of fn_wrapper
    fn_wrapper.__name__ = fn.__name__
    return register_model(fn_wrapper)


@register_tiny_vit_model
def tiny_vit_5m_224(pretrained=False, num_classes=1000, drop_path_rate=0.0):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_11m_224(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_224(pretrained=False, num_classes=2884, drop_path_rate=0.2): # 改 1月6号
    return TinyViT(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_384(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=384,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_512(pretrained=False, num_classes=2884, drop_path_rate=0.1):
    return TinyViT(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
    )


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        weight = self.weight.float()
        support = torch.matmul(input, weight.cuda())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


'''Laplacian Matrix transorm'''


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class RelativeCoordPredictor(nn.Module):  # 根据输入特征图 x 计算相对坐标、相对距离、相对角度等信息，并生成位置权重矩阵。
    def __init__(self):
        super().__init__()

    def forward(self, x):    # 公式 7-11
        N, C, H, W = x.shape
        # print("00",x.shape)
        # print("!!!",H,W) # 14 14
        # reduced_x = x.view(N, C, H*W).transpose(1, 2).contiguous()  # (N, S, C)

        mask = torch.sum(x, dim=1)
        size = H

        # mask = mask.view(N, H * W)
        # thresholds = torch.mean(mask, dim=1, keepdim=True)
        # binary_mask = (mask > (thresholds*0.25)).float()
        # binary_mask = binary_mask.view(N, H, W)

        binary_mask = torch.ones(N, H, W).cuda()

        masked_x = x * binary_mask.view(N, 1, H, W)
        masked_x = masked_x.view(N, C, H * W).transpose(1, 2).contiguous()  # (N, S, C)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)

        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()
        #         basic_index = torch.from_numpy(np.array([i for i in range(N)]))

        # max_features = reduced_x[basic_index, reduced_x_max_index, :]  # (N, C)
        # max_features_to_concat = max_features.unsqueeze(1).expand((N, H*W, C))

        basic_label = torch.from_numpy(self.build_basic_label(size)).float()
        # Build Label
        label = basic_label.cuda()
        #         label = basic_label

        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H * W, 2)  # (N, S, 2)
        # label = label.type(torch.long)
        basic_anchor = label[basic_index.long(), reduced_x_max_index.long(), :].unsqueeze(1)  # (N, 1, 2)
        relative_coord = label - basic_anchor
        relative_coord = relative_coord / size
        relative_dist = torch.sqrt(torch.sum(relative_coord ** 2, dim=-1))  # (N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)

        binary_relative_mask = binary_mask.view(N, H * W)
        relative_dist = relative_dist * binary_relative_mask
        relative_angle = relative_angle * binary_relative_mask

        basic_anchor = basic_anchor.squeeze(1)  # (N, 2)

        relative_coord_total = torch.cat((relative_dist.unsqueeze(2), relative_angle.unsqueeze(2)), dim=-1)

        '''# Calc Loss
        preds_dist, preds_angle = preds_coord[:, :, :, 0], preds_coord[:, :, :, 1]

        preds_dist = preds_dist.view(N, H, W)
        relative_dist = relative_dist.view(N, H, W)
        dist_loss = self.dist_loss_f(preds_dist, relative_dist)

        preds_angle = preds_angle.view(N, H*W)
        gap_angle = preds_angle - relative_angle  # (N, S) in (0, 1) - (0, 1) = (-1, 1)
        gap_angle[gap_angle < 0] += 1
        gap_angle = gap_angle - torch.mean(gap_angle, dim=-1, keepdim=True)  # (N, H*W)
        gap_angle = gap_angle.view(N, H, W)
        angle_loss = torch.pow(gap_angle, 2)
        '''

        position_weight = torch.mean(masked_x, dim=-1)
        position_weight = position_weight.unsqueeze(2)
        position_weight = torch.matmul(position_weight, position_weight.transpose(1, 2))

        # relative_coord_total = relative_coord_total.half()
        # position_weight = position_weight.half()

        return relative_coord_total, basic_anchor, position_weight, reduced_x_max_index
    # 返回相对坐标信息 relative_coord_total、基准锚点 basic_anchor、位置权重矩阵 position_weight 和最大索引 reduced_x_max_index
    def build_basic_label(self, size):
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label
    # 生成一个大小（size × size）的二维numpy数组

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x  # （N，nclass）


class Part_Structure(nn.Module):
    def __init__(self, config=None):
        super(Part_Structure, self).__init__()

        ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.act_fn = ACT2FN["relu"]
        self.dropout = Dropout(0.1)
        self.relative_coord_predictor = RelativeCoordPredictor()

        self.gcn = GCN(2, 2, 576, dropout=0.1)

    def forward(self, hidden_states, attention_map):
    # def forward(self, hidden_states, attention_map, debug=False, save_path="debug_viz.png"):
        B, C, H, W = attention_map.shape
        structure_info, basic_anchor, position_weight, reduced_x_max_index = self.relative_coord_predictor(
            attention_map)
        # structure_info 相对坐标信息 结点

        # 🟢 控制是否输出可视化
        # if debug:
        #     visualize_relative_info(attention_map, self.relative_coord_predictor, batch_index=0, save_path=save_path)


        structure_info = self.gcn(structure_info, position_weight)

        out = torch.ones((B, 576)).cuda()
        for i in range(B):
            index = int(basic_anchor[i, 0] * H + basic_anchor[i, 1])

            # hidden_states[i, 0] = hidden_states[i, 0] + structure_info[i, index, :]

            out[i] = structure_info[i, index, :]

        # return structure_info
        return out      # （B, 576)

# net = tiny_vit_21m_224(pretrained=False)
# input = torch.ones((2,3,448,448))
# net(input)