import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from PIL import Image
from random import randint


import cv2
import numpy as np

import cfg
import model4
from model4 import NetD


args = cfg.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4)#一次训练所取的样本数
parser.add_argument('--imageSize', type=int, default=128)#图片的大小,这个训练集里面图片的大小都是一致的
parser.add_argument('--nz', type=int, default=args.latent_dim, help='size of the latent z vector')#暂时不知道
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')#训练的轮数
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')#学习率
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')#参数
parser.add_argument('--data_path', default='data1/', help='folder to train data')
parser.add_argument('--data_path2', default='data2/', help='folder to train data')
parser.add_argument('--outf', default='myImgs2/', help='folder to output images and model checkpoints')
opt = parser.parse_args()

#图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize),
    torchvision.transforms.CenterCrop(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])#使用平均值和标准偏差对张量图像进行规格化,消除量纲

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
dataset2 = torchvision.datasets.ImageFolder(opt.data_path2, transform=transforms)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

dataloader = torch.utils.data.DataLoader(#数据取器
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    drop_last=True,
)

dataloader2 = torch.utils.data.DataLoader(#数据取器
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    drop_last=True,
)

netG = model4.Generator().to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.001, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0



for epoch in range(1, opt.epoch + 1):
    for i, (img) in enumerate(zip(dataloader,dataloader2)):

        #images1是原图
        #images2是线稿图
        # print("size:")
        # print(imgs.shape)
        # print(type(imgs))
        # imgs = torch.squeeze(imgs)  # 若不标注删除第几维度，则会删除所有为1的维度
        # plt.imshow(imgs.reshape(opt.imageSize,opt.imageSize,3))
        # plt.imshow(imgs.numpy().reshape(opt.imageSize,opt.imageSize,3))

        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()          #把模型中参数的梯度设为0
        ## 让D尽可能的把真图片判别为1 ,这个是真图
        imgs=img[1][0].to(device)
        output = netD(imgs)
        output = torch.squeeze(output)#squeeze去掉维数为1的的维度 unsqueeze对维度进行扩充
        label.data.fill_(real_label)
        label=label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()

        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        #noise = torch.randn(opt.batchSize, opt.nz)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        noise=img[0][0].to(device)
        '''
            gpu版本
        '''
        with torch.no_grad(): #避免梯度传到G，因为G不用更新
            fake = netG(noise)  # 生成假图
        output = netD(fake)
        '''
            cpu版本
        '''
        # test process
        #output = netD(fake.detach()) #避免梯度传到G，因为G不用更新
        output = torch.squeeze(output)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        #之前的fake 是不能传播梯度的，这里一定要重新生成
        fake = netG(noise)  # 生成假图
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        output = torch.squeeze(output)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    # torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
    # torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))






