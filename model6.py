import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.signal
from torch.nn import AvgPool2d

import cfg

'''
    在model5的基础上，更换损失函数（从image-to-image论文上发现的）为L1，
    训练的时候需要每次打乱dataloder的顺序才有用吗？
    调整了drop_rate 参数，从0->0.1 减少原图的占比
'''
# 矩阵相乘
class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x




def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)
class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 防止过拟合
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        ##################################################################
        # if head_dim == 0:
        #     head_dim = 0.1
        self.scale = qk_scale or head_dim ** -0.5

        # 弄出3个矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 防止过拟合
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # 矩阵相乘
        self.mat = matmul()
        self.is_mask = is_mask


    def forward(self, x):

        #例如： X = [-1,16,64]
        B, N, C = x.shape
        if self.is_mask == 1:
            H = W = int(math.sqrt(N))
            image = x.view(B, H, W, C).view(B * H, W, C)
            qkv = self.qkv(image).reshape(B * H, W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1, 2)
            x = x.reshape(B * H, W, C).view(B, H, W, C).view(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.is_mask == 2:
            H = W = int(math.sqrt(N))
            # （B*W，H，C）
            image = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B * W, H, C)

            # （B*W，H，C * 3） -> (B*W,H,3,num_heads,C//num_heads)->(3,B*W,self.num_heads,H, C // self.num_heads)
            qkv = self.qkv(image).reshape(B * W, H, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1, 2).reshape(B * W, H, C).view(B, W, H, C).permute(0, 2, 1, 3).reshape(B,
                                                                                                                    N,
                                                                                                                    C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu):
        super().__init__()
        # 归一化
        self.norm1 = nn.LayerNorm(dim)
        # 计算注意力
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here 一些加减乘除 和dropout一样防止结果过拟合
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 归一化
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer="gelu"):
        super().__init__()
        self.depth = depth
        self.block = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
            ) for i in range(depth)])

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x


def pixel_upsample(x, H, W):
    '''

    :param x:[B,N,C]
    :param H: -> 2H
    :param W: -> 2W
    :return:x:[B,N*4,C/4]
    '''
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    #[-1,C,H,W] -> [-1,C/4,H*2,W*2]
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W

def bicubic_upsample(x, H, W):
    '''

    :param x: [B,N,C]
    :param H: 2H
    :param W: 2W
    :return: x: [B,4*N,C]
    '''
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        num_windows / 64
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Generator(nn.Module):
    def __init__(self,num_heads=4,mlp_ratio=4.,qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,):
        super(Generator,self).__init__()

        #每一层block 的 attention 数量
        self.depth = depth = (5,4,2,1)

        self.pos_drop = nn.Dropout(p=0.4)

        self.embed_dim  = 64
        self.window_size = 4

        self.pos_embed_0 = nn.Parameter(torch.zeros(1, 4, 128))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 16, 64))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 64, 32))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 256, 16))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, 1024, 32))
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, 4096, 16))
        self.pos_embed_6 = nn.Parameter(torch.zeros(1, 16384, 4))

        self.pos_embed = [
            self.pos_embed_0,
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
            self.pos_embed_5,
            self.pos_embed_6,
        ]



        self.blocks = nn.ModuleList([
            StageBlock(
                depth=self.depth[1],
                dim=128,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ),
            StageBlock(
                depth=self.depth[1],
                dim=64,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ),
            StageBlock(
                depth=self.depth[1],
                dim=32,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ),
            StageBlock(
                depth=self.depth[1],
                dim=16,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ),
            StageBlock(
                depth=self.depth[1],
                dim=32,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ), StageBlock(
                depth=self.depth[1],
                dim=16,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ),StageBlock(
                depth=self.depth[1],
                dim=4,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0
            ),
        ])

        self.tRGB =  nn.Conv2d(4, 3, 1, 1, 0)

        self.layer0 = nn.Sequential(
            nn.Conv2d(32, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(8 , 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )


        #[-1,3,128,128] -> [-1,3,16,16]  -> [128,2,2]
        self.fRGB_0 = nn.Conv2d(3,128,kernel_size=5,stride=3,padding=0,dilation=3,)
        #[3,128,128] -> [3,32,32] -> [32,4,4]
        self.fRGB_1 = nn.Conv2d(3,32,kernel_size=5,stride=5,padding=0,dilation=4,)
        # [3,128,128] -> [3,64,64] -> [16,8,8]
        self.fRGB_2 = nn.Conv2d(3,16,kernel_size=8,stride=5,padding=0,dilation=4,)

        self.fRGB_3 = nn.Conv2d(3,8,kernel_size=14,stride=5,padding=0,dilation=4,)
        #
        self.fRGB_4 = nn.Conv2d(3, 16, kernel_size=10, stride=3, padding=1, dilation=4, )
        self.fRGB_5 = nn.Conv2d(3, 8, kernel_size=9, stride=1, padding=0, dilation=8, )



        # layer5输出尺寸 3x96x96
        self.convlayer = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):

        # x:[-1,3,128,128]
       # print(self.layer5(x).view(-1,8,4096).size())
        #x_5 :
        x_5 = self.layer5(x).view(-1,8,4096).permute(0,2,1)
        x_4 = self.layer4(x_5.permute(0,2,1).view(-1,8,64,64)).view(-1,16,1024).permute(0,2,1)
        x_3 = self.layer3(x_4.permute(0,2,1).view(-1,16,32,32)).view(-1,8,256).permute(0,2,1)
        x_2 = self.layer2(x_3.permute(0,2,1).view(-1,8,16,16)).view(-1,16,64).permute(0,2,1)
        x_1 = self.layer1(x_2.permute(0,2,1).view(-1,16,8,8)).view(-1,32,16).permute(0,2,1)
        x_0 = self.layer0(x_1.permute(0,2,1).view(-1,32,4,4)).view(-1,128,4).permute(0,2,1)



        #[-1,4,128]
        y = x_0 + self.pos_embed[0]
        #y = self.pos_drop(y)
        H, W = 2,2
        y = self.blocks[0](y)

        # [-1,4,128] -> [-1,16,32]
        y, H, W = pixel_upsample(y, H, W)
        # [-1,16,32] -> [-1,16,64]
        y = torch.cat([y,x_1],dim=-1) + self.pos_embed[1]
        y = self.blocks[1](y)

        # [-1,16,64] -> [-1,64,16]
        y, H, W = pixel_upsample(y, H, W)
        # [-1,64,16] -> [-1,64,32]
        y = torch.cat([y, x_2], dim=-1) + self.pos_embed[2]
        y = self.blocks[2](y)

        #[-1,64,32] -> [-1,256,8]
        y, H, W = pixel_upsample(y, H, W)
        y = torch.cat([y, x_3], dim=-1) + self.pos_embed[3]
        y = self.blocks[3](y)

        # [-1,256,16] -> [-1,1024,16]
        y, H, W = bicubic_upsample(y, H, W)
        # [-1,1024,16] ->[-1,1024,32]
        y = torch.cat([y, x_4], dim=-1) + self.pos_embed[4]
        y = self.blocks[4](y)


        # [-1,64,32] -> [-1,256,8]
        y, H, W = pixel_upsample(y, H, W)
        y = torch.cat([y, x_5], dim=-1) + self.pos_embed[5]
        #block里面是(64*64*16)^2 需要的内存太大了，这时候就需要用到分块的思想，计算每一块之间的注意力  之前是y = self.blocks[5](y)
        B, _, C = y.size()
        y = y.view(B, H, W, C)
        y = window_partition(y, self.window_size)
        y = y.view(-1, self.window_size * self.window_size, C)
        y = self.blocks[5](y)
        y = y.view(-1, self.window_size, self.window_size, C)
        #[-1,4096,16]
        y = window_reverse(y, self.window_size, H, W).view(B, H * W, C)

        # #[-1,128*128,4]
        y, H, W = pixel_upsample(y, H, W)
        y = y + self.pos_embed[6]
        B, _, C = y.size()
        y = y.view(B, H, W, C)
        y = window_partition(y, self.window_size)
        y = y.view(-1, self.window_size * self.window_size, C)
        y = self.blocks[6](y)
        y = y.view(-1, self.window_size, self.window_size, C)
        # [-1,4096,16]
        y = window_reverse(y, self.window_size, H, W).view(B, H * W, C)


        output = self.tRGB(y.permute(0, 2, 1).view(B, 4, H, W))

        return output

# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # 定义NetD的前向传播
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out



